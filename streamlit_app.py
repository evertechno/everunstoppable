import streamlit as st
import torch
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForCTC
import nltk
from nltk.tokenize import word_tokenize
from io import BytesIO
from pydub import AudioSegment

# Download necessary NLTK corpora at runtime (this is the only external request needed)
nltk.download('punkt')
nltk.download('stopwords')

# Load the Wav2Vec2 model for ASR
@st.cache_resource
def load_asr_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return processor, model

# Load the sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    sentiment_model = pipeline("sentiment-analysis")  # Automatically downloads model from HuggingFace
    return sentiment_model

# Convert MP3 to WAV using pydub
def mp3_to_wav(uploaded_file):
    audio = AudioSegment.from_mp3(uploaded_file)
    wav_file = BytesIO()
    audio.export(wav_file, format="wav")
    wav_file.seek(0)
    return wav_file

# Transcribe audio using Wav2Vec2
def transcribe_audio(uploaded_file):
    processor, model = load_asr_model()
    
    # Convert MP3 to WAV format
    wav_file = mp3_to_wav(uploaded_file)

    # Load audio as numpy array
    import numpy as np
    audio_input = np.frombuffer(wav_file.read(), np.int16)

    # Use processor to convert audio into the correct input format for Wav2Vec2
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
    logits = model(input_values=inputs.input_values).logits

    # Decode the predicted ids to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]

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
