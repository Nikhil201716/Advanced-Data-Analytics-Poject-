import matplotlib.pyplot as plt
import numpy as np
import cv2
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """
    Cleans text data by removing punctuation, converting to lowercase, removing stopwords, and lemmatizing the words.
    """
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Convert words to lower case and split them
    text = text.lower().split()

    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]

    # Lemmatize words
    text = [WordNetLemmatizer().lemmatize(word) for word in text]

    # Join the words back into one string separated by space
    text = " ".join(text)
    return text

def plot_image(image, title="Image"):
    """
    Plots a single image using Matplotlib.
    """
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_and_resize_image(image_path, size=(128, 128)):
    """
    Loads an image from the disk and resizes it to the specified size.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, size)
    return image

def feature_standardization(features):
    """
    Standardizes features by subtracting the mean and dividing by the standard deviation.
    """
    features = np.array(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features_standardized = (features - mean) / std
    return features_standardized
