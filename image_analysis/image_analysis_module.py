import streamlit as st
import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale and resizes it.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        st.error(f"Error loading image {image_path}.")
        return None
    # Resize image to reduce dimensionality
    image = cv2.resize(image, (100, 100))
    return image.flatten()  # Flatten the image to create a feature vector

def image_analysis_function(image_folder):
    """
    Analyze images in the specified folder using a simple machine learning model.
    """
    if not os.path.exists(image_folder):
        st.error("The specified folder does not exist.")
        return

    # Load all images
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_paths:
        st.error("No images found in the specified folder.")
        return

    st.write(f"Performing analysis on {len(image_paths)} images.")

    # Process images and extract features
    features = [load_and_preprocess_image(path) for path in image_paths if load_and_preprocess_image(path) is not None]

    # Dummy labels for demonstration (e.g., classify images based on filename)
    labels = [1 if 'cat' in path.lower() else 0 for path in image_paths]  # Example: Classify images as 'cat' or not

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

    # Train a SVM classifier
    classifier = SVC(gamma='auto')
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    # Evaluation
    report = classification_report(y_test, predictions)
    st.text(report)

    # Optionally, visualize the first image and its PCA transformation
    if features:
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(features)
        plt.figure()
        plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=labels)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Images')
        st.pyplot(plt)

# Example usage in Streamlit
if st.button('Run Image Analysis'):
    image_folder_path = st.text_input('Enter the path to the image folder:')
    if image_folder_path:
        image_analysis_function(image_folder_path)
