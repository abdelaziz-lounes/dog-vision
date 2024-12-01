import tensorflow as tf
import tensorflow_hub as hub
import os
import datetime

# Model parameters
INPUT_SHAPE = (224, 224, 3)
OUTPUT_SHAPE = 120  # Assuming 120 unique breeds in the dataset
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"
NUM_EPOCHS = 10

def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    """
    Builds a transfer learning model with TensorFlow Hub.
    """
    feature_extractor_layer = hub.KerasLayer(model_url, trainable=False, input_shape=input_shape)
    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense(output_shape, activation="softmax")
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    return model

def create_tensorboard_callback():
    """
    Creates a TensorBoard callback for logging.
    """
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)

def train_model(X_train, y_train, X_val, y_val, epochs=NUM_EPOCHS):
    """
    Trains a model and saves it.
    """
    model = create_model()
    tensorboard_cb = create_tensorboard_callback()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[tensorboard_cb]
    )
    model.save(os.path.join("models", "dog_breed_model.h5"))
    print("Model trained and saved successfully.")
    return model
