import streamlit as st
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import google.generativeai as genai
import numpy as np
import nltk

# Download necessary NLTK corpora at runtime
nltk.download('punkt')
nltk.download('stopwords')

# Configure the Google Gemini API key from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Load the Wav2Vec2 model for ASR
@st.cache_resource
def load_asr_model():
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        return processor, model
    except Exception as e:
        st.error(f"Error loading ASR model: {e}")
        return None, None

# Transcribe audio using Wav2Vec2
def transcribe_audio(uploaded_file):
    try:
        processor, model = load_asr_model()

        if processor is None or model is None:
            return "Error loading model."

        # Load audio as numpy array
        audio_input = np.frombuffer(uploaded_file.read(), np.int16)

        # Use processor to convert audio into the correct input format for Wav2Vec2
        inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
        logits = model(input_values=inputs.input_values).logits

        # Decode the predicted ids to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)

        return transcription[0]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return "Error during transcription."

# Function to analyze transcription using Google Gemini AI
def analyze_transcription_with_ai(transcription_text):
    try:
        # Generate a prompt for the AI to analyze the transcription
        analysis_prompt = f"Analyze this customer support call transcription: {transcription_text}. Provide sentiment, tone, and keywords from the text."

        # Generate response (analysis) from Gemini AI
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(analysis_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return "Error during analysis."

# Streamlit app UI
def main():
    st.title("Ever AI - Call Transcription and Analysis")
    st.write("Upload your customer support call recording (WAV format only) for transcription and analysis.")

    # File uploader for WAV files only
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded_file is not None:
        st.write(f"File uploaded: {uploaded_file.name}")

        # Transcribe the uploaded audio file
        st.subheader("Transcription")
        transcription = transcribe_audio(uploaded_file)
        st.write(transcription)

        if transcription != "Error during transcription.":
            # Analyze the transcription using Gemini AI
            st.subheader("Analysis of Transcription")
            analysis_result = analyze_transcription_with_ai(transcription)
            st.write(analysis_result)

if __name__ == "__main__":
    main()
