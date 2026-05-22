import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OBJECT_NAMES, OBJECT_MATERIALS, SURFACE_DICT, EXTRA_ELEMENTS


def get_exp_description(event, scene, object_name, material_desc, surface, view):
    """Build the event-specific part of the generation prompt."""
    movement = "rolling"

    if event == "fall":
        return (f"The video shows a {view} view of a {material_desc} {object_name} "
                f"smoothly {movement} across a {surface} in a {scene}. "
                f"When the {object_name} reaches the edge, it falls off the {surface} "
                f"and vertically descends until it hits the ground. "
                f"The object maintains its shape and does not break during the fall.")

    elif event == "collision":
        extra = EXTRA_ELEMENTS["collision"][scene]
        return (f"The video shows a {view} view of a {material_desc} {object_name} "
                f"smoothly {movement} across a {surface}, colliding with a {extra} "
                f"on a {scene}. "
                f"The objects react to the collision but preserve their rigidity "
                f"and do not break.")

    elif event == "occlusion":
        extra = EXTRA_ELEMENTS["occlusion"][scene]
        return (f"The video shows a {view} view of a {material_desc} {object_name} "
                f"smoothly {movement} and passing behind a {extra} on a {surface} "
                f"in a {scene}. "
                f"The {object_name} becomes temporarily occluded by the {extra} "
                f"before reappearing. "
                f"The {extra} remains stationary throughout the sequence.")

    raise ValueError(f"Unknown event: {event!r}")


_SUFFIX = (
    "Everything on the video follows the natural behaviour of solid objects "
    "in a physical environment. Objects do not fly, morph or disappear. "
    "No new elements appear on the scene. "
    "The background is static. Fixed camera view, no camera movement."
)


def get_prompt(event, scene, obj, appearance, view, generic_object=False):
    """Build the generation prompt for one CRONOS experiment.

    Parameters
    ----------
    event      : "fall" | "collision" | "occlusion"
    scene      : "kitchen" | "house" | "apartment" | "garden" | "pool"
    obj        : object key from OBJECT_NAMES (e.g. "TennisBall")
    appearance : material variant, e.g. "material0", "material1", "material2"
    view       : camera view string, e.g. "front", "side", "top", "close"
    generic_object : if True, replace the specific object name with "object"
    """
    object_name   = "object" if generic_object else OBJECT_NAMES[obj]
    material_desc = OBJECT_MATERIALS[obj][int(appearance[-1])]
    surface       = SURFACE_DICT[event][scene]

    description = get_exp_description(event, scene, object_name, material_desc, surface, view)
    return description + " " + _SUFFIX


def main():
    parser = argparse.ArgumentParser(
        description="Generate a text prompt for a CRONOS experiment.",
    )
    parser.add_argument("--event",      required=True,
                        choices=["fall", "collision", "occlusion"])
    parser.add_argument("--scene",      required=True,
                        choices=["kitchen", "house", "apartment", "garden", "pool"])
    parser.add_argument("--object",     required=True, choices=list(OBJECT_NAMES.keys()))
    parser.add_argument("--appearance", required=True,
                        help="Material variant, e.g. material0, material1, material2")
    parser.add_argument("--view",       required=True,
                        help="Camera view, e.g. front, side, top, close")
    parser.add_argument("--generic-object", action="store_true",
                        help="Use 'object' instead of the specific object name")
    parser.add_argument("--output", default=None,
                        help="Write prompt to this file instead of stdout")
    args = parser.parse_args()

    prompt = get_prompt(
        event=args.event,
        scene=args.scene,
        obj=args.object,
        appearance=args.appearance,
        view=args.view,
        generic_object=args.generic_object,
    )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(prompt + "\n")
        print(f"Prompt saved → {args.output}")
    else:
        print(prompt)


if __name__ == "__main__":
    main()
