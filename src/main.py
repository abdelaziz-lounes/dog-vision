from pretraitement import preprocess_data, create_data_batches
from train import train_model
from predict import predict
import os

DATA_DIR = "data"
MODEL_PATH = os.path.join("models", "dog_breed_model.h5")

# Preprocess data
X_train, X_val, y_train, y_val, unique_breeds = preprocess_data(DATA_DIR)

# Create batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)

# Train model
model = train_model(train_data, y_train, val_data, y_val)

# Predict on new data
test_images = ["data/test/image1.jpg", "data/test/image2.jpg"]
predicted_labels = predict(MODEL_PATH, test_images, unique_breeds)

print("Predicted labels:", predicted_labels)