import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    QWEN_MODEL_PATH, QWEN_FPS, QWEN_MIN_PIXELS, QWEN_MAX_PIXELS,
    QWEN_MAX_NEW_TOKENS, QWEN_TEMPERATURE, QWEN_TOP_P, QWEN_TOP_K,
    OBJECT_NAMES, SURFACE_DICT, EXTRA_ELEMENTS,
)

_CONTEXT = (
    "You are evaluating a short video of a physics simulation. "
    "An object in the scene was pushed just before the video starts. "
    "Watch the video carefully and answer all of the following questions about what you observe. "
    "For each question provide a True or False answer. "
    "Also provide brief comments justifying your answers and rate your overall confidence "
    "from 0.0 (not confident at all) to 1.0 (fully confident)."
)


def _event_context(event, scene, obj):
    name    = OBJECT_NAMES[obj]
    surface = SURFACE_DICT[event][scene]
    if event == "fall":
        return (f"The video should show a {name} rolling across a {surface} "
                f"in a {scene} and falling off the edge.")
    elif event == "collision":
        coll = EXTRA_ELEMENTS["collision"][scene]
        return (f"The video should show a {name} rolling across a {surface} "
                f"in a {scene} and colliding with a {coll}.")
    elif event == "occlusion":
        occ = EXTRA_ELEMENTS["occlusion"][scene]
        return (f"The video should show a {name} rolling across a {surface} "
                f"in a {scene} and passing behind a {occ}.")
    raise ValueError(f"Unknown event: {event}")


def _get_questions(event, scene, obj):
    """Return (questions list, positive_outcomes dict) for the given experiment."""
    name    = OBJECT_NAMES[obj]
    surface = SURFACE_DICT[event][scene]

    general_q = [
        "Is the camera view fixed throughout the video?",
        "Is the background static throughout the video?",
        f"Does the {name} maintain its color and shape throughout the video?",
        "Do new objects appear on the scene during the video?",
        "Do objects move smoothly without sudden jumps or teleportation during the video?",
        "Do objects move without forces acting on them?",
    ]
    general_pos = {
        "camera_fixed": True, "background_static": True, "object_consistent": True,
        "new_objects": False, "smooth_motion": True, "unexplained_motion": False,
    }

    if event == "fall":
        exp_q = [
            f"Does the {name} fall off the {surface} when it reaches the edge?",
            f"Does the {name} hit the ground?",
            f"Does the {name} change its direction while falling?",
            f"Does the {name} move on an arc-shaped path while falling?",
            f"Does the {name} accelerate while falling?",
        ]
        exp_pos = {
            "falls_off_surface": True, "hits_ground": True,
            "changes_direction": False, "arc_path": True, "accelerates": True,
        }
    elif event == "collision":
        coll = EXTRA_ELEMENTS["collision"][scene]
        exp_q = [
            f"Does the {name} contact with the {coll}?",
            f"Does the {name} come to a stop before contacting the {coll}?",
            f"Is the motion of {name} affected by the collision with the {coll}?",
            "Is the reaction to the impact realistic considering the size and mass of the objects?",
            "Are objects deformed or broken during or after the collision?",
        ]
        exp_pos = {
            "collision_occurs": True, "object_stops_before": False,
            "motion_affected_by_collision": True, "realistic_reaction": True,
            "deformation_or_breaking": False,
        }
    elif event == "occlusion":
        occ = EXTRA_ELEMENTS["occlusion"][scene]
        exp_q = [
            f"Does the {name} move behind the {occ} during the video?",
            f"Does the {name} reappear on the other side after being occluded by the {occ}?",
            f"Does the {name} move on a straight path during the whole video?",
            f"Does the {name} disappear after being occluded by the {occ}?",
            f"Does the {name} change its appearance after being occluded by the {occ}?",
        ]
        exp_pos = {
            "occlusion_occurs": True, "reappears_after_occlusion": True,
            "straight_path": True, "disappears_after_occlusion": False, "appearance_change": False,
        }
    else:
        raise ValueError(f"Unknown event: {event}")

    questions, outcomes = general_q + exp_q, {**general_pos, **exp_pos}
    # Questions are zipped against the outcome keys, so a length mismatch silently
    # pairs every later key with the wrong question.
    assert len(questions) == len(outcomes), (len(questions), len(outcomes))
    return questions, outcomes


def _build_prompt(questions, positive_outcomes, event, scene, obj):
    lines = [_CONTEXT, "", _event_context(event, scene, obj), "", "Questions:"]
    for i, (key, q) in enumerate(zip(positive_outcomes, questions), 1):
        lines.append(f"  {i}. [{key}] {q}")
    answer_fields = "\n".join(f'    "{k}": true or false' for k in positive_outcomes)
    lines += [
        "",
        "Respond ONLY with valid JSON using exactly this structure (no extra text outside the JSON):",
        "{", '  "answers": {', answer_fields + ",", '    ...', "  },",
        '  "comments": "<brief justification for each answer>",',
        '  "confidence": <float 0.0–1.0>', "}",
    ]
    return "\n".join(lines)


def _parse_response(text, positive_outcomes):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            for k, v in data.get("answers", {}).items():
                if isinstance(v, str):
                    data["answers"][k] = v.lower() in ("true", "yes")
            return data
        except json.JSONDecodeError:
            pass
    return {"answers": {k: None for k in positive_outcomes}, "comments": text, "confidence": None}


def load_model(model_path=QWEN_MODEL_PATH):
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model.eval(), processor


def compute_predictions(model, processor, video_path, event, scene, obj):
    """Run VLM evaluation on a single video.

    Returns:
        dict with keys: answers, comments, confidence, raw,
                        questions, positive_outcomes, positive_fraction
    """
    questions, positive_outcomes = _get_questions(event, scene, obj)
    prompt_text = _build_prompt(questions, positive_outcomes, event, scene, obj)

    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "fps": QWEN_FPS,
         "min_pixels": QWEN_MIN_PIXELS, "max_pixels": QWEN_MAX_PIXELS},
        {"type": "text", "text": prompt_text},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, video_inputs_raw, video_kwargs = process_vision_info(
        messages, image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True, return_video_metadata=True,
    )
    # unpack (tensor, metadata_dict) tuples produced by return_video_metadata=True
    video_inputs = [v for v, _ in video_inputs_raw]
    video_kwargs["video_metadata"] = [m for _, m in video_inputs_raw]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(
        text=[text], videos=video_inputs,
        return_tensors="pt", **video_kwargs,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            temperature=QWEN_TEMPERATURE,
            top_p=QWEN_TOP_P,
            top_k=QWEN_TOP_K,
            do_sample=True,
        )

    input_len = inputs["input_ids"].shape[1]
    raw = processor.batch_decode(
        output_ids[:, input_len:], skip_special_tokens=True
    )[0].split("</think>")[-1].strip()

    result = _parse_response(raw, positive_outcomes)
    result["raw"] = raw
    result["questions"] = questions
    result["positive_outcomes"] = positive_outcomes

    answers = result.get("answers", {})
    positives = [answers.get(k) == v for k, v in positive_outcomes.items()
                 if answers.get(k) is not None]
    result["positive_fraction"] = sum(positives) / len(positives) if positives else float("nan")
    return result


def save_predictions(result, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL evaluation on a video.")
    parser.add_argument("--video",    required=True, help="Input .mp4 file")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json for the reference sequence")
    parser.add_argument("--output",   required=True, help="Output .json path")
    args = parser.parse_args()

    with open(args.metadata) as f:
        meta = json.load(f)
    event  = meta["event"]
    scene  = meta["scene"]
    obj    = meta["object"]

    llm, processor = load_model()
    result = compute_predictions(llm, processor, args.video, event, scene, obj)
    save_predictions(result, args.output)
    print(f"Positive fraction: {result['positive_fraction']:.3f}  →  {args.output}")


if __name__ == "__main__":
    main()
