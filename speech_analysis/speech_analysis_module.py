import streamlit as st
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import os

def load_audio_file(file_path):
    """
    Load an audio file and return the audio time series and its sampling rate.
    """
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_mfcc_features(y, sr, n_mfcc=13):
    """
    Extract MFCC features from an audio file.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)  # Scaling the MFCCs by taking the mean across columns
    return mfccs_scaled

def speech_analysis_function(audio_folder):
    """
    Analyze audio files in the specified folder using MFCC features and SVM classification.
    """
    if not os.path.exists(audio_folder):
        st.error("The specified folder does not exist.")
        return

    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3'))]
    if not audio_files:
        st.error("No audio files found in the specified folder.")
        return

    st.write(f"Performing analysis on {len(audio_files)} audio files.")

    features = []
    labels = []  # Dummy labels, here we assume the label is part of the filename before an underscore

    for file_path in audio_files:
        y, sr = load_audio_file(file_path)
        mfccs_scaled = extract_mfcc_features(y, sr)
        features.append(mfccs_scaled)
        label = file_path.split('_')[0]  # Extracting label from filename
        labels.append(label)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM Classifier
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    # Display results
    report = classification_report(y_test, predictions)
    st.text(report)

# Example usage within Streamlit
if st.button('Run Speech Analysis'):
    audio_folder_path = st.text_input('Enter the path to the audio folder:')
    if audio_folder_path:
        speech_analysis_function(audio_folder_path)
