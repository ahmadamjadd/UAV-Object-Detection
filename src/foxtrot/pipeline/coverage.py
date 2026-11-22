"""Parameter coverage report (V3): what happens to every setting.

Every key in the pinned Ultralytics ``cfg/default.yaml`` is classified as:

* ``exposed``      – the member can set it (a TrainingSettings field sends it);
* ``managed``      – Foxtrot sets it so jobs never overwrite each other or lose identity;
* ``inapplicable`` – belongs to another mode or task (predict, export, segmentation, ...);
* ``unsupported``  – applies to detection training but is not offered yet, with the reason.

Generator settings are listed the same way. A test fails if a runtime key is
missing here, or if an "exposed" key is not really sent to training.
"""

from foxtrot.config.training import TrainingSettings

EXPOSED, MANAGED, INAPPLICABLE, UNSUPPORTED = "exposed", "managed", "inapplicable", "unsupported"

# Training keys Foxtrot always sets itself, and why.
_MANAGED = {
    "task": "Always 'detect': Foxtrot datasets contain bounding boxes",
    "mode": "Always 'train' for training jobs",
    "model": "Chosen from the model registry and the job's start mode",
    "data": "Each job writes its own data.yaml from the registered dataset",
    "project": "Outputs always go to the job's own folder",
    "name": "Fixed per job so outputs never collide",
    "exist_ok": "Always False: a job never reuses another output folder",
    "pretrained": "Follows the job's start mode (pretrained/scratch/checkpoint)",
    "resume": "Interrupted-job recovery is planned for V7",
    "save": "Always True: the best checkpoint is required",
    "val": "Always True: best.pt is chosen by validation mAP",
    "split": "Evaluation always uses the dataset's validation split",
    "plots": "Off: plots are not uploaded",
    "verbose": "Off: progress is reported through Foxtrot events",
    "cls_remap": "Class mapping comes from the dataset's own validated classes",
}

# Detection-training keys not offered yet, with the reason.
_UNSUPPORTED = {
    "time": "A time limit would conflict with the epoch-based progress and results",
    "save_period": "Only the best checkpoint is kept and uploaded",
    "single_cls": "Classes come from the dataset's validated mapping",
    "rect": "Rectangular batches disable mosaic and change the evaluation; not verified",
    "profile": "Developer profiling output; not useful in the website",
    "multi_scale": "Changes image size during training; not verified on CPU",
    "compile": "torch.compile is not verified on the supported CPU setup",
    "channels_last": "Memory-layout tuning for GPU; not verified",
    "nbs": "Nominal batch size for loss scaling; advanced, not exposed yet",
    "conf": "Evaluation thresholds stay fixed so runs are comparable",
    "iou": "Evaluation thresholds stay fixed so runs are comparable",
    "max_det": "Evaluation thresholds stay fixed so runs are comparable",
    "half": "Prediction/export option; training precision is set by 'amp'",
    "save_json": "COCO JSON output is not used",
}

# Keys for other modes (predict/export/track) or other tasks (segment, pose, obb, classify, depth).
_INAPPLICABLE = (
    "overlap_mask mask_ratio dropout quantize dnn nms source vid_stride stream_buffer visualize augment "
    "agnostic_nms classes retina_masks embed show save_frames save_txt save_conf save_crop show_labels "
    "show_conf show_boxes line_width format optimize dynamic simplify opset workspace distill_model dis "
    "cls_pw pose kobj rle angle dlog dgrad dlam copy_paste copy_paste_mode auto_augment erasing cfg tracker "
    "int8 keras batch_size"
).split()


# The Ultralytics keys that TrainingSettings fields send (from the fields' metadata).
def exposed_keys() -> dict[str, str]:
    keys = {}
    for name, field in TrainingSettings.model_fields.items():
        key = (field.json_schema_extra or {}).get("ultralytics")
        if key:
            keys[key] = name
    return keys


# Classify one runtime key.
def classify(key: str) -> tuple[str, str]:
    exposed = exposed_keys()
    if key in exposed:
        return EXPOSED, f"Training setting '{exposed[key]}'"
    if key in _MANAGED:
        return MANAGED, _MANAGED[key]
    if key in _UNSUPPORTED:
        return UNSUPPORTED, _UNSUPPORTED[key]
    if key in _INAPPLICABLE:
        return INAPPLICABLE, "Used by another mode or task, not detection training"
    return UNSUPPORTED, "Not classified yet (new in this Ultralytics version)"


# Original generator options (params.yaml and command line), and how V2 handles each.
GENERATOR_COVERAGE = {
    "n": (EXPOSED, "recipe.images_per_class"),
    "size": (EXPOSED, "recipe.canvas_size"),
    "shape_scale.s0/s1": (EXPOSED, "recipe.shape_scale"),
    "noise": (EXPOSED, "recipe.noise"),
    "blur.b0/b1": (EXPOSED, "recipe.blur"),
    "brightness_diff": (EXPOSED, "recipe.brightness_difference"),
    "contrast.alpha0/alpha1": (EXPOSED, "recipe.contrast"),
    "brightness.beta0/beta1": (EXPOSED, "recipe.brightness"),
    "saturation.sat0/sat1": (EXPOSED, "recipe.saturation"),
    "desaturate": (EXPOSED, "recipe.desaturate"),
    "letter.add_letter / letter_prob": (EXPOSED, "recipe.add_letter, recipe.letter_probability"),
    "shadow.add_light_gradient / blend_prob / opacity": (
        EXPOSED, "recipe.add_light_gradient, recipe.light_probability, recipe.light_opacity"),
    "seed (fixed 42 in the script)": (EXPOSED, "recipe.seed"),
    "--border": (UNSUPPORTED, "Command-line only; borders change box geometry and are not verified"),
    "--alt_blur / --alt_blur_prob": (UNSUPPORTED, "Command-line only; not part of params.yaml or verified"),
    "--letter_abs_rot_prob": (UNSUPPORTED, "Command-line only; not part of params.yaml or verified"),
    "--letter_dataset / --only_letter": (INAPPLICABLE, "Letter-crop datasets need their own task contract"),
    "--display / --save_bounds / --show_distribution / --freq / --debug": (
        INAPPLICABLE, "Terminal viewing and debugging only"),
    "shapes (class_names.py)": (MANAGED, "Shared class list; changing it changes label meaning"),
}


# The full report for the website and docs: runtime keys plus generator settings.
def coverage_report(runtime_keys) -> dict:
    runtime = [{"key": key, "status": status, "detail": detail}
               for key in runtime_keys for status, detail in [classify(key)]]
    generator = [{"key": key, "status": status, "detail": detail}
                 for key, (status, detail) in GENERATOR_COVERAGE.items()]
    return {"runtime": runtime, "generator": generator}


# Keys of the installed Ultralytics default configuration, read without importing Ultralytics.
def ultralytics_default_keys() -> list[str]:
    import importlib.util
    from pathlib import Path

    import yaml

    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:
        return []
    path = Path(spec.origin).parent / "cfg" / "default.yaml"
    return list(yaml.safe_load(path.read_text(encoding="utf-8")).keys())
