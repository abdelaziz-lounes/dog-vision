import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_keras
import matplotlib.pyplot as plt

class DogBreedModel:
    def __init__(self, model_path, unique_breeds):
        self.unique_breeds = unique_breeds
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """Loads the trained model from the .h5 file."""
        model = tf_keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        return model
    
    def predict(self, image_path):
        """Makes a prediction on a single image and plots the image with the label."""
        image = self._preprocess_image(image_path)
        prediction = self.model.predict(tf.expand_dims(image, axis=0))
        predicted_label = self.unique_breeds[np.argmax(prediction)]

        # Plot the image and predicted label
        plt.imshow(image)
        plt.title(f"Predicted Label: {predicted_label}")
        plt.axis("off")
        plt.show()  # Display the plot

        return predicted_label

    def _preprocess_image(self, image_path):
        """Preprocesses the image for the model."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=[224, 224])  # IMG_SIZE is 224
        return image