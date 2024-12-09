import streamlit as st
import torch
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
import speech_recognition as sr

# Download necessary NLTK corpora at runtime (this is the only external request needed)
nltk.download('punkt')
nltk.download('stopwords')

# Load sentiment analysis model from Hugging Face
@st.cache_resource
def load_sentiment_model():
    sentiment_model = pipeline("sentiment-analysis")  # Automatically downloads model from HuggingFace
    return sentiment_model

# Transcribe audio using Google Web Speech API
def transcribe_audio(uploaded_file):
    recognizer = sr.Recognizer()
    
    # Convert the uploaded file to audio format supported by SpeechRecognition
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with sr.AudioFile("temp_audio.wav") as source:
        audio = recognizer.record(source)
        
        try:
            # Using Google Web Speech API for transcription
            transcription = recognizer.recognize_google(audio)
            return transcription
        except sr.UnknownValueError:
            return "Sorry, could not understand the audio."
        except sr.RequestError:
            return "Sorry, there was an error with the speech recognition service."
    # Cleanup after transcription
    os.remove("temp_audio.wav")

# Analyze sentiment and keywords
def analyze_transcription(text):
    sentiment_model = load_sentiment_model()
    sentiment = sentiment_model(text)
    sentiment_score = sentiment[0]["label"]
    sentiment_confidence = sentiment[0]["score"]

    # Simple keyword extraction using NLTK
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = word_tokenize(text)
    keywords = [word for word in word_tokens if word.lower() not in stop_words and word.isalpha()]

    return sentiment_score, sentiment_confidence, keywords

# Streamlit app
def main():
    st.title("Customer Support Call Transcription and Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a Customer Support Call Recording", type=["mp3", "wav", "flac"])

    if uploaded_file is not None:
        st.write(f"File uploaded: {uploaded_file.name}")
        
        # Transcribe audio file
        st.subheader("Transcription")
        transcription = transcribe_audio(uploaded_file)
        st.write(transcription)

        # Sentiment analysis and keyword extraction
        st.subheader("Sentiment Analysis and Keywords")
        sentiment, confidence, keywords = analyze_transcription(transcription)
        st.write(f"Sentiment: {sentiment} (Confidence: {confidence*100:.2f}%)")
        st.write(f"Extracted Keywords: {', '.join(keywords)}")

        # Support metrics evaluation
        st.subheader("Support Metrics Evaluation")
        if sentiment == "POSITIVE":
            st.write("The call is likely to be a positive support interaction.")
        elif sentiment == "NEGATIVE":
            st.write("The call indicates a negative support experience.")
        else:
            st.write("Neutral sentiment detected in the call.")

if __name__ == "__main__":
    main()
