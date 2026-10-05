import cv2
import os

def resize_images(input_directory, output_directory, target_width=256, target_height=256):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image
            img = cv2.imread(os.path.join(input_directory, filename))
            
            if img is not None:
                resized_img = cv2.resize(img, (target_width, target_height))
                cv2.imwrite(os.path.join(output_directory, filename), resized_img)

            else:
                print(f"Failed to read {filename}")

if __name__ == "__main__":
    input_dir = "../Backgrounds/Backgrounds"
    output_dir = "../Backgrounds/Backgrounds256"
    resize_images(input_dir, output_dir)
