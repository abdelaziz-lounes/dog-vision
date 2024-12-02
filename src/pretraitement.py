import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Define image size
IMG_SIZE = 224

def process_image(image_path):
    """
    Reads an image file path and preprocesses it into a Tensor.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

def get_image_label(image_path, label):
    """
    Returns a tuple of (processed_image, label).
    """
    image = process_image(image_path)
    return image, label

def create_data_batches(X, y=None, batch_size=32, valid_data=False, test_data=False):
    """
    Creates batches of image (X) and label (y) pairs for training/validation/testing.
    """
    if test_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = data.map(process_image).batch(batch_size)
        return data_batch

    elif valid_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = data.map(get_image_label).batch(batch_size)
        return data_batch

    else:  # Training data
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.map(get_image_label).batch(batch_size)
        return data_batch

def preprocess_data(data_dir):
    """
    Loads and preprocesses the data from the given directory.
    """
    labels_csv = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    filenames = [os.path.join(data_dir, "train", fname + ".jpg") for fname in labels_csv["id"]]
    labels = labels_csv["breed"].to_numpy()

    # Encode labels to integers
    unique_breeds = np.unique(labels)
    label_to_index = {breed: i for i, breed in enumerate(unique_breeds)}
    labels = np.array([label_to_index[label] for label in labels])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(filenames, labels, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, unique_breeds

