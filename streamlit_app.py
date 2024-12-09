import streamlit as st
import torch
from transformers import pipeline
from coqui.ai import Model
import nltk
from nltk.tokenize import word_tokenize
import os

# Download necessary NLTK corpora at runtime (this is the only external request needed)
nltk.download('punkt')
nltk.download('stopwords')

# Load ASR model from Coqui
@st.cache_resource
def load_asr_model():
    model = Model.from_pretrained("coqui-ai/coop_stt_en")  # Download this model automatically
    return model

# Load sentiment analysis model from Hugging Face
@st.cache_resource
def load_sentiment_model():
    sentiment_model = pipeline("sentiment-analysis")  # Automatically downloads model from HuggingFace
    return sentiment_model

# Transcribe audio
def transcribe_audio(uploaded_file):
    model = load_asr_model()
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    transcription = model.transcribe("temp_audio.wav")
    os.remove("temp_audio.wav")  # Cleanup after transcription
    return transcription["text"]

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
