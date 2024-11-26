import os
from PIL import Image

# Dataset path
DATASET_PATH = r"D:\data (1)"

def is_image_file(file_path):
    """Check if the file is a valid image."""
    try:
        img = Image.open(file_path)
        img.verify()  # Verify the integrity
        return True
    except (IOError, SyntaxError):
        return False

def clean_dataset():
    """Remove or log invalid image files."""
    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            if not is_image_file(file_path):
                print(f"Invalid image file: {file_path}")
                os.remove(file_path)  # Remove invalid file

if __name__ == "__main__":
    clean_dataset()
    print("Dataset cleaned successfully!")
