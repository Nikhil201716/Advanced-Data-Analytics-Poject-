import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
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

def social_media_analysis_function(df, text_column='text'):
    """
    Performs social media analysis including preprocessing, TF-IDF vectorization, sentiment analysis, and training a Naive Bayes model.

    Parameters:
        df (DataFrame): DataFrame containing the social media text data.
        text_column (str): Column name containing the text.
    """
    if text_column not in df.columns:
        st.error("Specified text column does not exist in the DataFrame.")
        return

    # Preprocess text data
    st.write("Preprocessing text data...")
    df[text_column] = df[text_column].astype(str).apply(preprocess_text)
    
    # Display cleaned data
    st.write("Preview of cleaned text data:")
    st.dataframe(df[[text_column]].head())

    # Vectorization
    st.write("Vectorizing text data...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df[text_column])

    # Sentiment analysis
    st.write("Performing sentiment analysis...")
    df['sentiment'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
    st.write("Preview of sentiment analysis results:")
    st.dataframe(df[['sentiment']].head())

    # Visualize sentiment distribution
    st.write("Sentiment Distribution:")
    st.bar_chart(df['sentiment'])

    if 'label' in df.columns:
        y = df['label']
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a Naive Bayes classifier
        classifier = MultinomialNB()
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        # Display results
        st.write("Model Performance:")
        st.text(f"Accuracy: {accuracy_score(y_test, predictions)}")
        st.text(f"Classification Report:\n{classification_report(y_test, predictions)}")
    else:
        st.write("No label column provided for training the model.")