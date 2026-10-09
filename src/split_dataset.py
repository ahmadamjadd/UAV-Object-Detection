import os
import yaml
import shutil
import numpy as np

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def main():
    # 1. Load parameters
    with open(os.path.join(project_root, "params.yaml"), "r") as f:
        params = yaml.safe_load(f)

    split_params = params.get("split", {"train": 0.80, "val": 0.20})
    train_ratio = split_params["train"]
    val_ratio = split_params["val"]

    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio}"

    # 2. Set seed for deterministic split
    np.random.seed(42)

    # 3. Get list of images
    generated_dir = os.path.join(project_root, "data", "generated")
    images_dir = os.path.join(generated_dir, "images")
    labels_dir = os.path.join(generated_dir, "labels")

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    n_total = len(image_files)

    print(f"Found {n_total} images to split (train={train_ratio}, val={val_ratio})")

    # 4. Shuffle and split
    indices = np.random.permutation(n_total)
    n_train = int(n_total * train_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    splits = {
        "train": train_indices,
        "val": val_indices,
    }

    # 5. Create output directory structure
    split_dir = os.path.join(project_root, "data", "split")

    # Clean previous split
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)

    for split_name in ["train", "val"]:
        os.makedirs(os.path.join(split_dir, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, split_name, "labels"), exist_ok=True)

    # 6. Copy files into split folders
    for split_name, split_indices in splits.items():
        for idx in split_indices:
            img_file = image_files[idx]
            label_file = os.path.splitext(img_file)[0] + ".txt"

            # Copy image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(split_dir, split_name, "images", img_file)
            shutil.copy2(src_img, dst_img)

            # Copy label
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(split_dir, split_name, "labels", label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

        print(f"  {split_name}: {len(split_indices)} images")

    print(f"Split complete. Output at: {split_dir}")

if __name__ == "__main__":
    main()
