import os
import sys
import glob
from ultralytics import YOLO

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def main():
    test_images_dir = os.path.join(project_root, "data", "test_real", "images")
    predictions_dir = os.path.join(project_root, "data", "test_real", "predictions")

    # 1. Check if test images exist
    if not os.path.exists(test_images_dir):
        print("No test_real/images/ directory found. Skipping inference.")
        return

    image_files = glob.glob(os.path.join(test_images_dir, "*.jpg")) + \
                  glob.glob(os.path.join(test_images_dir, "*.png")) + \
                  glob.glob(os.path.join(test_images_dir, "*.jpeg"))

    if len(image_files) == 0:
        print("No test images found in data/test_real/images/. Skipping inference.")
        return

    print(f"Found {len(image_files)} test images. Running inference...")

    # 2. Find the best model weights
    # Look for best.pt in the runs directory
    best_model_path = None
    runs_dir = os.path.join(project_root, "runs")
    
    for root, dirs, files in os.walk(runs_dir):
        if "best.pt" in files:
            best_model_path = os.path.join(root, "best.pt")
            break

    if best_model_path is None:
        print("ERROR: No best.pt model found in runs/. Train a model first.")
        sys.exit(1)

    print(f"Using model: {best_model_path}")

    # 3. Load model and run inference
    model = YOLO(best_model_path)

    # Clean previous predictions
    if os.path.exists(predictions_dir):
        import shutil
        shutil.rmtree(predictions_dir)
    os.makedirs(predictions_dir, exist_ok=True)

    # Run inference and save annotated images
    results = model.predict(
        source=test_images_dir,
        save=True,
        save_txt=True,
        project=predictions_dir,
        name=".",
        exist_ok=True,
        conf=0.25
    )

    print(f"Inference complete. {len(image_files)} images processed.")
    print(f"Annotated images saved to: {predictions_dir}")

if __name__ == "__main__":
    main()
