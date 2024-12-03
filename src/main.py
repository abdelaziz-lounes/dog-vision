import os
import pandas as pd
import tensorflow as tf
from model import DogBreedModel  # Import DogBreedModel

class Main:
    def __init__(self, model_path, labels_csv_path):
        self.unique_breeds = self._extract_unique_breeds(labels_csv_path)
        self.model = DogBreedModel(model_path, self.unique_breeds)  # Instantiate DogBreedModel

    def _extract_unique_breeds(self, labels_csv_path):
        """Extracts unique breeds from the labels.csv file."""
        labels_df = pd.read_csv(labels_csv_path)
        unique_breeds = labels_df["breed"].unique()
        return unique_breeds

    def run(self, image_paths):
        """Makes predictions on a list of image paths."""
        for image_path in image_paths:
            prediction = self.model.predict(image_path)  # Call predict on the model instance
            print(f"Prediction for {image_path}: {prediction}")

if __name__ == "__main__":
    # Define paths
    model_path = os.path.join("models", "saved-model-all-images-Adam.h5")  # Replace with your model's .h5 file path
    labels_csv_path = os.path.join("data", "labels.csv")
    image_dir = os.path.join( "data", "dogsftest")  # Path to your images folder

    main = Main(model_path, labels_csv_path)
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith((".jpg", ".jpeg", ".png"))]
    main.run(image_paths)