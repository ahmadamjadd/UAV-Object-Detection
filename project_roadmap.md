# UAV Object Detection MLOps Roadmap

## 🎯 Project Objective
To build a highly professional, reproducible, and automated MLOps pipeline for **Team Foxtrot (GIKI)**. 
The pipeline will handle everything from synthetic dataset generation to training YOLO models, tracking experiments, and exporting the final model for deployment on a Jetson Orin Nano.

## 🛠️ Core Technologies
- **Git & GitHub**: Source code version control.
- **DVC (Data Version Control)**: Large file and dataset versioning.
- **DagsHub**: Remote storage for DVC datasets and built-in MLflow tracking server.
- **MLflow**: Experiment tracking (logging hyperparameters, metrics, and models).
- **Google Colab**: Free GPU compute for model training.
- **YOLOv8**: State-of-the-art object detection framework.

---

## 🗺️ The Step-by-Step Plan

### Phase 1: Foundation & Determinism
**Goal**: Make the current dataset generator robust and reproducible.
- [x] **Step 1: Explore & Understand** - Analyze the existing OpenCV dataset generator.
- [x] **Step 2: Configuration** - Move hardcoded command-line arguments into a centralized `params.yaml` file.
- [x] **Step 3: Determinism** - Seed the random number generators so the exact same dataset can be recreated infinitely as long as `params.yaml` is unchanged.

### Phase 2: Version Control
**Goal**: Separate lightweight code from heavy data.
- [x] **Step 4: DVC & DagsHub Setup** - Initialize Git for code. Initialize DVC for data. Connect both to DagsHub so large datasets don't crash GitHub.

### Phase 3: Automation & Tracking
**Goal**: Automate the sequence of scripts and track results.
- [x] **Step 5: DVC Pipeline (`dvc.yaml`)** - Link the scripts together. If `params.yaml` changes, DVC automatically knows it needs to re-run the generator.
- [x] **Step 6: MLflow Integration** - Add YOLO training (`src/train.py`) to the pipeline and configure it to securely stream training metrics (accuracy, loss) to DagsHub's MLflow server.

### Phase 4: Cloud Training & Validation
**Goal**: Move the heavy lifting to the cloud and make validation meaningful.
- [ ] **Step 7A: Google Colab Workflow** - Create a Jupyter Notebook (`notebooks/train_on_colab.ipynb`) that can:
  1. Clone the GitHub repo.
  2. Pull the heavy dataset from DagsHub via `dvc pull`.
  3. Train the YOLO model on a free T4 GPU.
  4. Automatically push the newly trained model weights back to DagsHub via `dvc push`.
- [x] **Step 7B: Dataset Splitting & Real-World Testing** - Make validation metrics meaningful and enable testing on real flight images:
  1. Add a `split_dataset` stage (`src/split_dataset.py`) that deterministically splits `data/generated/` into `data/split/train/` and `data/split/val/` so the model no longer validates on its own training data.
  2. Update `src/train.py` to point YOLO at `data/split/` with separate train/val paths.
  3. Add a `test_model` stage (`src/test.py`) that runs inference on real flight images in `data/test_real/images/` and saves annotated predictions. Gracefully skips if no test images are present.
  4. Updated DVC pipeline: `generate_dataset --> split_dataset --> train_model --> test_model`.

### Phase 5: Hardware Deployment
**Goal**: Prepare the model for the actual UAV hardware.
- [ ] **Step 8: Jetson Export** - Write a deployment script (`src/export.py`) that takes the best YOLO `.pt` model and converts it into a highly optimized format (like ONNX or TensorRT) specifically tuned for the Jetson Orin Nano.

### Phase 6: Continuous Integration (Optional)
**Goal**: Make the robots do the work.
- [ ] **Step 9: CI/CD Pipeline** - Write GitHub Actions so that whenever a team member changes `params.yaml` and pushes to GitHub, a cloud server automatically wakes up, generates the new dataset, trains the model, and uploads the results.
