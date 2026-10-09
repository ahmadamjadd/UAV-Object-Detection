import os
import yaml
import sys
from ultralytics import YOLO

# Add the data generation folder to path so we can import the SHAPES list
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(project_root, "src", "data_generation"))
from DataSetGeneration import SHAPES

def create_yolo_yaml(dataset_dir):
    """Creates the data.yaml file required by YOLOv8"""
    data_yaml_path = os.path.join(project_root, "data.yaml")
    
    # For now, we will use the same images for train and val since we haven't split them yet.
    # We map the classes directly from the SHAPES list used in the generator!
    yolo_data = {
        "path": dataset_dir,
        "train": "images",
        "val": "images",
        "names": {i: shape for i, shape in enumerate(SHAPES)}
    }
    
    with open(data_yaml_path, "w") as f:
        yaml.dump(yolo_data, f)
        
    return data_yaml_path

def main():
    # 1. Load hyper-parameters
    with open(os.path.join(project_root, "params.yaml"), "r") as f:
        params = yaml.safe_load(f)
    
    train_params = params.get("train", {"epochs": 3, "batch_size": 16})
    
    # 2. Configure MLflow to securely log to DagsHub
    # YOLOv8 automatically detects these environment variables and logs EVERYTHING for you!
    os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/muhammadahmadamjad0/UAV-Object-Detection.mlflow"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "YOLO_Synthetic_Training"
    
    # You MUST set these locally in your terminal or Colab before running the script:
    # export MLFLOW_TRACKING_USERNAME=muhammadahmadamjad0
    # export MLFLOW_TRACKING_PASSWORD=<YOUR_DAGSHUB_TOKEN>
    
    # 3. Create data.yaml
    dataset_dir = os.path.join(project_root, "data", "generated")
    data_yaml_path = create_yolo_yaml(dataset_dir)
    
    # 4. Train the YOLOv8 model!
    print(f"Starting YOLO Training for {train_params['epochs']} epochs...")
    
    model = YOLO("yolov8n.pt") # Load the nano version of YOLOv8
    
    results = model.train(
        data=data_yaml_path,
        epochs=train_params["epochs"],
        batch=train_params["batch_size"],
        project="runs",
        name="synthetic_train"
    )
    
    # 5. Log dataset generation parameters to the same MLflow run
    import mlflow
    
    # Get the run that YOLO just created
    run = mlflow.last_active_run()
    if run:
        data_params = params.get("data_generation", {})
        # Flatten nested params so they show as e.g. "data/n", "data/noise"
        flat_params = {}
        for key, value in data_params.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_params[f"data/{key}/{sub_key}"] = sub_value
            else:
                flat_params[f"data/{key}"] = value
        
        with mlflow.start_run(run_id=run.info.run_id):
            mlflow.log_params(flat_params)
        
        print("Dataset parameters logged to MLflow.")

if __name__ == "__main__":
    main()
