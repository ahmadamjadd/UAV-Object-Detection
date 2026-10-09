# MLOps Pipeline Journey

## Project Purpose
The goal of this project is to build a reproducible MLOps pipeline for Team Foxtrot (GIKI) to train object detection models for UAV competitions. We are starting with a parameterized synthetic dataset generator (using OpenCV) that pastes various shapes and letters onto backgrounds. The MLOps pipeline will allow team members to seamlessly change dataset parameters, generate data, train a YOLO model, evaluate it on a fixed real-world test set, and export the model (ONNX/TensorRT) for deployment on a Jetson Orin Nano. 

We are using DVC for data/model versioning, MLflow for experiment tracking, and DagsHub as the remote server. Training will be done on Google Colab.

---

## Step 1: Explore Repo and Summarize Generator
**Status**: Completed

**What we did**: 
Explored the repository to understand how the synthetic dataset is generated.
- Identified that `DataSetGeneration.py` orchestrates the creation of images, while `Shapes.py` handles drawing specific geometries.
- Documented the key parameters the generator accepts via the terminal (e.g., `--size`, `--n`, `--noise`, `--b0`, `--alpha0`, etc.). These parameters control the shape dimensions, augmentations (blur, noise, brightness, contrast), and letter placements.
- Identified that currently, these parameters are passed directly via the command line, which makes tracking experiments difficult. Moving these to a configuration file is the necessary next step for reproducibility.

## Step 2: Propose Repo Structure and Create params.yaml
**Status**: Completed

**What we did**:
- Discussed the proposed folder structure for the repository to cleanly separate code, data, and configs.
- Created `params.yaml` at the root of the project. This file holds the dataset generation parameters (size, noise, blur, etc.). DVC will track this file to know when hyperparameters change and when to re-run the pipeline.

**Proposed Directory Structure:**
```text
.
├── data/
│   ├── raw/                 # Downloaded backgrounds (tracked by DVC, not Git)
│   ├── generated/           # Output of our generator script
│   └── test/                # The fixed real-world test set
├── src/                     
│   ├── data_generation/     # We will move DataSetGeneration.py and Shapes.py here
│   ├── train.py             
│   ├── evaluate.py          
│   └── deploy.py            
├── params.yaml              # Hyperparameters
├── dvc.yaml                 # DVC pipeline stages
└── notebooks/               # For Colab
```

## Step 3: Deterministic Data Generation and `params.yaml` Integration
**Status**: Completed

**What we did**:
- Created the new directory structure and moved the dataset generation code to `src/data_generation/` and the background images to `data/raw/`.
- Updated `DataSetGeneration.py` to read hyperparameters directly from `params.yaml` using the `yaml` library, replacing the old command-line arguments.
- Added a fixed random seed (`seed = 42`) for `numpy`, `random`, and `cv2` inside the generator. This guarantees that the exact same datasets can be recreated at any time simply by keeping the `params.yaml` unchanged, saving precious storage space while maintaining reproducibility.

## Step 4: Setup DVC and DagsHub Remote
**Status**: Completed

**What we did**:
- Created a `.gitignore` to prevent Git from uploading the large `data/` and `venv/` folders.
- Initialized a Git repository and successfully pushed the codebase to both DagsHub and GitHub.
- Initialized DVC (Data Version Control) inside the repository.
- Added DagsHub as the DVC remote storage (`dvc remote add origin...`). This setup allows us to version control our large datasets. The heavy dataset files will go to DagsHub via DVC, while lightweight pointers and code go to Git.

## Step 5: Creating the DVC Pipeline (`dvc.yaml`)
**Status**: Completed

**What we did**:
- Created a `dvc.yaml` file to define our MLOps pipeline.
- Added a `generate_dataset` stage that runs the synthetic dataset generator script.
- Configured DVC to track dependencies (`src/data_generation/` and `data/raw/`) and outputs (`data/generated/`).
- Linked the `data_generation` parameters from `params.yaml` so DVC knows exactly when hyperparameters have changed and when a re-run is necessary.

## Step 6: Add MLflow Logging
**Status**: Completed

**What we did**:
- Installed `mlflow`, `dagshub`, and `ultralytics`.
- Wrote `src/train.py`, which dynamically generates YOLO's `data.yaml` based on the classes used in `DataSetGeneration.py`.
- Linked `src/train.py` to DagsHub's MLflow server using environment variables (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`).
- Added a `train_model` stage to `dvc.yaml` and a `train` parameter block to `params.yaml`, ensuring that the training stage is tracked by DVC just like the dataset generation stage.

## Step 7B: Dataset Splitting & Real-World Testing
**Status**: Completed

**What we did**:
- **Problem identified**: `train.py` was pointing both `train` and `val` to the same `data/generated/images/` folder, meaning the model was validating on its own training data and all reported validation metrics (mAP50, precision, recall) were meaningless.
- **Created `src/split_dataset.py`**: A new script that reads split ratios from `params.yaml` (`split.train: 0.80`, `split.val: 0.20`), deterministically shuffles the generated images (seeded with `np.random.seed(42)`), and copies them into `data/split/train/` and `data/split/val/` with separate `images/` and `labels/` subdirectories. The original `data/generated/` stays intact.
- **Added `split` section to `params.yaml`**: Defines train/val ratios so DVC tracks when they change.
- **Updated `src/train.py`**: Changed `create_yolo_yaml()` to point YOLO at `data/split/` with separate `train: train/images` and `val: val/images` paths, so validation metrics now reflect performance on unseen data.
- **Created `src/test.py`**: A script that checks for real flight images in `data/test_real/images/`. If images are present, it loads the best trained model from `runs/`, runs inference, and saves annotated predictions to `data/test_real/predictions/`. If no test images are found, it prints a skip message and exits cleanly (creating the empty predictions directory so DVC does not error on missing outputs).
- **Created `data/test_real/images/`**: An empty directory (with `.gitkeep`) where team members can manually drop real flight photos for testing.
- **Updated `dvc.yaml`**: Added `split_dataset` stage (between `generate_dataset` and `train_model`) and `test_model` stage (after `train_model`). The full pipeline is now: `generate_dataset --> split_dataset --> train_model --> test_model`.
- **Bug fix**: The `test_model` stage in `dvc.yaml` originally declared `data/test_real/predictions/` as an output. When no test images exist, the script skips inference and never creates that directory, causing DVC to fail with an "output does not exist" error. Fixed by removing the `outs` declaration from the `test_model` stage entirely -- predictions only exist when there are actual test images to run on, so DVC shouldn't track a directory that may or may not exist.
