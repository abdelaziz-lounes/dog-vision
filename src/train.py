import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
import os
import datetime

# Model parameters
INPUT_SHAPE = (224, 224, 3)
OUTPUT_SHAPE = 120  # 120 unique breeds in the dataset
MODEL_URL = "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/130-224-classification/2"
NUM_EPOCHS = 50

def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    """
    Builds a transfer learning model with TensorFlow Hub.
    """
    feature_extractor_layer = hub.KerasLayer(model_url, trainable=False, input_shape=input_shape)
    model = tf_keras.Sequential([
        feature_extractor_layer,
        tf_keras.layers.Dense(output_shape, activation="softmax")
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf_keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    return model

def create_tensorboard_callback():
    """
    Creates a TensorBoard callback for logging.
    """
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf_keras.callbacks.TensorBoard(log_dir=log_dir)

def train_model(train_data, val_data, epochs=NUM_EPOCHS):
    """
    Trains a model and saves it.
    """
    model = create_model()
    tensorboard_cb = create_tensorboard_callback()
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)  
    train_dataset = train_dataset.batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
    val_dataset = val_dataset.batch(32)
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[tensorboard_cb]
    )
    model.save(os.path.join("..","models", "dog_breed_model.h5"))
    print("Model trained and saved successfully.")
    return model
