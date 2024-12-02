import tensorflow as tf
import numpy as np
import tf_keras

def process_image(image_path, img_size=224):
    """
    Processes a single image for prediction.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_size, img_size])
    return image

def get_pred_label(prediction_probabilities, unique_breeds):
    """
    Converts prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]

def predict(model_path, image_paths, unique_breeds):
    """
    Loads a trained model and predicts labels for the given images.
    """
    model = tf_keras.models.load_model(model_path)
    images = [process_image(image_path) for image_path in image_paths]
    image_batch = tf.stack(images)  # Create a batch
    predictions = model.predict(image_batch)
    predicted_labels = [get_pred_label(pred, unique_breeds) for pred in predictions]
    return predicted_labels