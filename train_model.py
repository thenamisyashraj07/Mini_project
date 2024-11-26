import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import load_img, img_to_array

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dataset path
DATASET_PATH = r"D:\Hand_gesture_code\data (1)"
IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 32
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def is_valid_image(file_path):
    """Check if a file is a valid image."""
    try:
        load_img(file_path)  # Using Keras load_img to avoid PIL errors
        return True
    except Exception as e:
        print(f"Skipping invalid image {file_path}: {e}")
        return False

def filter_images(directory):
    """Filter out invalid images and prepare a clean file list."""
    valid_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file)[1].lower() in VALID_EXTENSIONS and is_valid_image(file_path):
                valid_files.append(file_path)
    return valid_files

def create_label_mapping(file_list):
    """Create a mapping from class names to integer labels."""
    class_names = sorted(set(os.path.basename(os.path.dirname(f)) for f in file_list))
    label_mapping = {name: idx for idx, name in enumerate(class_names)}
    return label_mapping

def custom_data_generator(file_list, label_mapping, batch_size, target_size):
    """Custom generator to yield batches of valid images and their labels."""
    while True:
        batch_images = []
        batch_labels = []

        for file_path in file_list:
            # Extract class label from the parent folder name and map it to an integer
            label_name = os.path.basename(os.path.dirname(file_path))
            label = label_mapping[label_name]

            # Load and preprocess the image
            img = load_img(file_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0

            batch_images.append(img_array)
            batch_labels.append(label)

            if len(batch_images) == batch_size:
                yield np.array(batch_images), np.array(batch_labels)
                batch_images, batch_labels = [], []

# Prepare the dataset
file_list = filter_images(DATASET_PATH)
label_mapping = create_label_mapping(file_list)

# Define the CNN model
model = Sequential([
    tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_mapping), activation='softmax')  # Using the number of classes from the mapping
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the custom generator
model.fit(
    custom_data_generator(file_list, label_mapping, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH)),
    steps_per_epoch=len(file_list) // BATCH_SIZE,
    epochs=10
)

# Save the model in the new Keras format
model.save("hand_gesture_model.keras")  # Changed to .keras format
print("Model saved as 'hand_gesture_model.keras'")
