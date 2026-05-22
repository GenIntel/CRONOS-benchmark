# ── Model paths ───────────────────────────────────────────────────────────────
SAM3_MODEL_PATH   = "/path/to/sam3"
SAM3D_OBJECTS_DIR = "/path/to/sam-3d-objects"
SAM3D_HF_REPO     = "facebook/sam-3d-objects"
SAM3D_HF_CACHE    = "/path/to/hf-cache/sam-3d-objects"
CLIP_MODEL_PATH   = "/path/to/CLIP"
QWEN_MODEL_PATH   = "Qwen/Qwen3-VL-32B-Instruct"

# ── Hub identifiers (repo, model) ─────────────────────────────────────────────
COTRACKER_HUB = ("facebookresearch/co-tracker", "cotracker3_offline")
DISMO_HUB     = ("CompVis/DisMo", "motion_extractor_large")
DINO_HUB      = ("facebookresearch/dinov2", "dinov2_vitl14")

# ── Inference defaults ────────────────────────────────────────────────────────
COTRACKER_GRID_SIZE = 50
DISMO_TIME_WINDOW   = 4
DINO_MIN_MASK_SIZE          = 16    # min bounding-box side (px) for a valid mask frame
MIN_OCCLUSION_FRACTION = 0.25  # exclude frames where visible object area < this fraction of frame-0 area
SAM3D_SEED          = 42
SAM3D_STRIDE        = 10   # process every N-th frame for 3D reconstruction

# ── Qwen VLM defaults ─────────────────────────────────────────────────────────
QWEN_FPS            = 1.0
QWEN_MIN_PIXELS     = 128 * 28 * 28
QWEN_MAX_PIXELS     = 512 * 28 * 28
QWEN_MAX_NEW_TOKENS = 1024
QWEN_TEMPERATURE    = 0.7
QWEN_TOP_P          = 0.8
QWEN_TOP_K          = 20

# ── VLM experiment metadata ───────────────────────────────────────────────────
OBJECT_NAMES = {
    "Can":        "can",
    "TennisBall": "tennis ball",
    "ToyTruck":   "toy SUV",
    "SoccerBall": "soccer ball",
    "Bottle":     "water bottle",
}

OBJECT_APPEARANCES = {
    "Can":        ["tin", "blue tin", "brown tin"],
    "TennisBall": ["green", "orange", "dark green"],
    "ToyTruck":   ["yellow", "red", "black"],
    "SoccerBall": ["white and black", "blue and black", "red and black"],
    "Bottle":     ["orange plastic", "red plastic", "yellow plastic"],
}

SURFACE_DICT = {
    "fall": {
        "kitchen":   "kitchen counter",
        "house":     "wooden table",
        "apartment": "wooden table",
        "garden":    "wooden table",
        "pool":      "wooden table",
    },
    "collision": {
        "kitchen":   "wooden table",
        "house":     "wooden table",
        "apartment": "wooden floor",
        "garden":    "wooden table",
        "pool":      "stone slabs",
    },
    "occlusion": {
        "kitchen":   "wooden table",
        "house":     "wooden table",
        "apartment": "wooden floor",
        "garden":    "wooden table",
        "pool":      "stone slabs",
    },
}

# ── Metric normalization ──────────────────────────────────────────────────────
# None → exponential decay (see METRIC_DECAY); explicit tuple → linear [lo, hi]
METRIC_RANGES = {
    "bg_mse":                  None,
    "motion_similarity":       (0.0, 1.0),
    "object_consistency":      (0.0, 1.0),
    "mean_chamfer_distance":   None,
    "vlm_positive_fraction":   (0.0, 1.0),
}
METRIC_DECAY = {
    "bg_mse":                50.0,
    "mean_chamfer_distance": 50.0,
}

# ── Success / failure thresholds ──────────────────────────────────────────────
# ("min", bound) → fail if below bound; ("max", bound) → fail if above bound
FAILURE_THRESHOLDS = {
    "bg_mse":                  ("max", 0.25),
    "motion_similarity":       ("min", 0.70),
    "object_consistency":      ("min", 0.50),
    "mean_chamfer_distance":   ("max", 0.80),
    "vlm_positive_fraction":   ("min", 0.35),
}

EXTRA_ELEMENTS = {
    "collision": {
        "house":     "blue ceramic cup",
        "apartment": "tennis ball",
        "garden":    "ceramic flower pot",
        "pool":      "ceramic jar",
        "kitchen":   "metallic vase",
    },
    "occlusion": {
        "house":     "ceramic flower pot",
        "apartment": "flower pot",
        "garden":    "clay flower pot",
        "pool":      "tree trunk",
        "kitchen":   "black water jar",
    },
}
