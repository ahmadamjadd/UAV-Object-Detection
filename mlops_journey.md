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

