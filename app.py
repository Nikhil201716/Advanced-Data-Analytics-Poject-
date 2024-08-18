import streamlit as st
import pandas as pd
from text_analysis.text_analysis_module import text_analysis_function
from image_analysis.image_analysis_module import image_analysis_function
from speech_analysis.speech_analysis_module import speech_analysis_function
from social_media_analysis.social_media_analysis_module import social_media_analysis_function

def main():
    st.title("Advanced Data Analysis Dashboard")
    
    # Sidebar for navigation
    analysis_type = st.sidebar.selectbox(
        "Choose the type of analysis:",
        ("Text Analysis", "Image Analysis", "Speech Analysis", "Social Media Analysis")
    )

    # Depending on the choice, we ask for different types of inputs
    if analysis_type == "Text Analysis" or analysis_type == "Social Media Analysis":
        uploaded_file = st.file_uploader("Upload your text file", type=['csv', 'txt'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_table(uploaded_file)

            st.write("Data Preview:", df.head())
            
            if analysis_type == "Text Analysis":
                text_analysis_function(df)
            elif analysis_type == "Social Media Analysis":
                social_media_analysis_function(df)

    elif analysis_type == "Image Analysis":
        # Allow user to input a folder path or choose files
        image_folder = st.text_input("Enter the folder path containing images:")
        if st.button("Run Image Analysis"):
            image_analysis_function(image_folder)

    elif analysis_type == "Speech Analysis":
        audio_folder = st.text_input("Enter the folder path containing audio files:")
        if st.button("Run Speech Analysis"):
            speech_analysis_function(audio_folder)

if __name__ == "__main__":
    main()
