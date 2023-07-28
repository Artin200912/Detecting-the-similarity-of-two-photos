import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load pre-trained VGG16 model without classification layers
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Function to extract features from the image
def extract_features(image_path):
    preprocessed_img = preprocess_image(image_path)
    features = model.predict(preprocessed_img)
    return features.flatten()

# Save features and labels to a database (simplified version, use a proper database in practice)
database = {}

def save_to_database(image_path, label):
    features = extract_features(image_path)
    database[label] = features

# Load a new image to compare
def compare_images(new_image_path):
    new_features = extract_features(new_image_path)
    similarities = {}
    
    for label, features in database.items():
        # Calculate cosine similarity (dot product) between new features and reference features
        similarity = np.dot(new_features, features) / (np.linalg.norm(new_features) * np.linalg.norm(features))
        similarities[label] = similarity
    
    # Find the label with the highest similarity
    most_similar_label = max(similarities, key=similarities.get)
    similarity_percentage = similarities[most_similar_label] * 100
    
    return most_similar_label, similarity_percentage

# Example usage
if __name__ == "__main__":
    reference_image_path = "jan.jpg"
    new_image_path = "car.jpg"
    label = "Your label for the reference image"
    
    # Save the reference image to the database
    save_to_database(reference_image_path, label)
    
    # Compare a new image with the reference image
    most_similar_label, similarity_percentage = compare_images(new_image_path)
    
    print(f"The new image is {similarity_percentage:.2f}% similar to the reference image labeled as '{most_similar_label}'")
