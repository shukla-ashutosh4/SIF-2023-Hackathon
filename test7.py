import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import cv2
import requests
from io import BytesIO

# Define image dimensions
height = 64
width = 64
channels = 3

# Function to load images and labels from the Bhuvan API source
def load_images_and_labels_from_api(api_url):
    # Make a request to the Bhuvan API to get image URLs and labels
    response = requests.get(api_url)
    data = response.json()  # Assuming the API returns data in JSON format

    # Print the structure of the JSON response
    print("JSON Response Structure:", data)

    # Extract image URLs and labels
    image_urls = [entry['image_url'] for entry in data]
    labels = [entry['label'] for entry in data]
    
    # Load images and preprocess
    images = []
    for url in image_urls:
        response = requests.get(url)
        img = cv2.imdecode(np.asarray(bytearray(response.content), dtype=np.uint8), 1)
        img = cv2.resize(img, (height, width))  # Resize images to a consistent size
        images.append(img)
    
    return np.array(images), np.array(labels)

# Define the Bhuvan API URL
bhuvan_api_url = 'https://bhuvan-app1.nrsc.gov.in/api/api_proximity/curl_hos_pos_prox.php?theme=hospital&lat=16.27939453125&lon=80.58837890625&buffer=3000&token=xxxxxxxxxxxxxx'

# Load image data and labels from the Bhuvan API
X, y = load_images_and_labels_from_api(bhuvan_api_url)

# Shuffle the data to ensure randomness
X, y = shuffle(X, y, random_state=42)

# Data preprocessing
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build a model with convolutional layers for image processing
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Model Accuracy: {accuracy * 100:.2f}%")
