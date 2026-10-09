# Step 7B: Dataset Splitting & Real-World Testing Plan

## Overview

Three changes to make validation metrics meaningful and enable testing on real flight images.

---

## Change 1: Split Dataset Stage (Train/Val Only)

**Problem:** Currently `train.py` points both `train` and `val` to the same `data/generated/images/` folder. The model validates on its own training data, so metrics like mAP50, precision, and recall are meaningless.

**Solution:**
- Add a `split` section to `params.yaml`:
  ```yaml
  split:
    train: 0.80
    val: 0.20
  ```
- Write `src/split_dataset.py`:
  - Reads split ratios from `params.yaml`
  - Takes `data/generated/` as input
  - Produces `data/split/` with this structure:
    ```
    data/split/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    ```
  - Files are **copied** (not moved) so `data/generated/` stays intact
  - Uses a seeded shuffle so the split is deterministic (same seed = same split)
- Add a new DVC pipeline stage between `generate_dataset` and `train_model`

**Result:** Validation metrics now reflect performance on **unseen** data.

---

## Change 2: Update `train.py` for Real Train/Val

**What changes:**
- `create_yolo_yaml()` updated to point to `data/split/` with separate paths:
  ```yaml
  path: data/split
  train: train/images
  val: val/images
  ```
- No other changes needed. YOLO automatically reports separate train and val metrics.

---

## Change 3: Test on Real Flight Images

**This is completely separate from the synthetic dataset.**

- Create `data/test_real/images/` folder. Team members manually drop real flight photos here whenever they want.
- Write `src/test.py`:
  - Checks if `data/test_real/images/` contains any images
  - If **empty or missing**: prints "No test images found, skipping" and exits cleanly
  - If **images exist**: loads the best model from `runs/`, runs inference, saves annotated output images to `data/test_real/predictions/`
- Add a `test_model` stage to `dvc.yaml` that runs **after** `train_model`
- `data/test_real/predictions/` tracked by DVC so results are versioned

---

## Updated Pipeline

```
generate_dataset --> split_dataset --> train_model --> test_model (skips if no images)
```

---

## Files to Create/Modify

| Action | File | Purpose |
|--------|------|---------|
| Create | `src/split_dataset.py` | Split generated data into train/val |
| Create | `src/test.py` | Run inference on real flight images |
| Modify | `params.yaml` | Add `split` section |
| Modify | `dvc.yaml` | Add `split_dataset` and `test_model` stages |
| Modify | `src/train.py` | Point to `data/split/` instead of `data/generated/` |
| Create | `data/test_real/images/` | Empty folder, team adds real photos manually |
