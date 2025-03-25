import os
import torch
import streamlit as st
import tempfile
import glob
from pathlib import Path
import asyncio
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter


st.set_page_config(page_title="Parrot TTS", layout="wide")
st.title("Parrot Text-to-Speech App")

@st.cache_resource
def setup_asyncio():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

setup_asyncio()

# Initialize directories
if 'initialized' not in st.session_state:
    os.makedirs('outputs_v3', exist_ok=True)
    os.makedirs('speaker_embeddings', exist_ok=True)
    os.makedirs('user_embeddings', exist_ok=True)  # Separate directory for user uploads
    os.makedirs('processed', exist_ok=True)
    st.session_state['initialized'] = True

# Model paths and device
@st.cache_resource
def load_models():
    ckpt_base = 'checkpoints_v3/base_speakers/EN'
    ckpt_converter = 'checkpoints_v3/converter'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = 'outputs_v3'
    
    try:
        base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
        base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')
        
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        
        return base_speaker_tts, tone_color_converter, device, output_dir, ckpt_base
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, device, 'outputs_v3', ckpt_base

# Load model
models = load_models()
if models[0] is not None:
    base_speaker_tts, tone_color_converter, device, output_dir, ckpt_base = models
else:
    st.stop()

# Get available accents - with exclusions
def get_available_accents():
    try:
        accent_files = glob.glob(f'{ckpt_base}/*.pth')
        # Exclude specific files
        excluded_files = ["checkpoint.pth", "jp.pth", "config.pth"]
        filtered_files = [file for file in accent_files if os.path.basename(file) not in excluded_files]
        accents = [os.path.splitext(os.path.basename(file))[0] for file in filtered_files]
        return accents
    except Exception as e:
        st.error(f"Error loading accents: {str(e)}")
        return ["en-us"]  # Default

accents = get_available_accents()

# Get available preset speaker embeddings
def get_available_preset_embeddings():
    try:
        embedding_files = glob.glob('speaker_embeddings/*_se.pth')
        embeddings = [os.path.basename(file).replace('_se.pth', '') for file in embedding_files]
        return embeddings
    except Exception:
        return []

# Get available user speaker embeddings
def get_available_user_embeddings():
    try:
        embedding_files = glob.glob('user_embeddings/*_se.pth')
        embeddings = [os.path.basename(file).replace('_se.pth', '') for file in embedding_files]
        return embeddings
    except Exception:
        return []

# Initialize session state for user uploaded files if not exists
if 'user_uploads' not in st.session_state:
    st.session_state['user_uploads'] = []

# Sidebar for model settings
st.sidebar.header("Voice Settings")

# Accent selection
accent = st.sidebar.selectbox("Select Accent", accents, index=accents.index('en-us') if 'en-us' in accents else 0)

# Text-to-speech settings
st.sidebar.subheader("TTS Settings")
speed = st.sidebar.slider("Speech Speed", 0.5, 1.5, 0.9, 0.1)
emotion = st.sidebar.selectbox("Emotion", ["default", "friendly", "cheerful", "excited", "sad", "angry", "terrified", "shouting", "whispering"], index=0)

# Main content
st.header("Text Input")
text_input = st.text_area("Enter the text you want to convert to speech:", 
                         "Hello, this is a demo of Parrot text-to-speech with voice cloning.")

# Voice selection
st.header("Voice Selection")
voice_option = st.radio("Choose voice option:", ["Use preset voice", "Use my uploaded voice"])

target_se = None

# Function to safely load embedding
def load_embedding(path):
    try:
        return torch.load(path).to(device)
    except Exception as e:
        st.error(f"Error loading voice embedding: {str(e)}")
        return None

if voice_option == "Use preset voice":
    preset_embeddings = get_available_preset_embeddings()
    if preset_embeddings:
        selected_preset = st.selectbox("Select Preset Voice", preset_embeddings)
        embedding_path = f'speaker_embeddings/{selected_preset}_se.pth'
        if os.path.exists(embedding_path):
            target_se = load_embedding(embedding_path)
    else:
        st.warning("No preset voices found.")

else:  # "Use uploaded voice"
    # Section for uploading new voices
    st.subheader("Upload New Voice")
    uploaded_file = st.file_uploader("Choose an audio file (MP3, WAV)", type=["mp3", "wav"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                reference_path = tmp_file.name
            
            extract_button = st.button("Extract Voice Characteristics")
            
            if extract_button:
                with st.spinner("Extracting voice characteristics..."):
                    try:
                        target_se, audio_name = se_extractor.get_se(reference_path, tone_color_converter, target_dir='processed', vad=True)
                        
                        # Save the embedding to user_embeddings folder
                        embedding_filename = Path(uploaded_file.name).stem
                        embedding_path = f'user_embeddings/{embedding_filename}_se.pth'
                        torch.save(target_se, embedding_path)
                        
                        # Add to session state
                        if embedding_filename not in st.session_state['user_uploads']:
                            st.session_state['user_uploads'].append(embedding_filename)
                        
                        st.success(f"Voice characteristics extracted and saved")
                        
                    except Exception as e:
                        st.error(f"Error extracting voice: {str(e)}")
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
    
    # Section for selecting from already uploaded voices
    st.subheader("Select from Your Uploaded Voices")
    user_embeddings = get_available_user_embeddings()
    
    if user_embeddings:
        selected_user_voice = st.selectbox("Select Your Voice", user_embeddings)
        embedding_path = f'user_embeddings/{selected_user_voice}_se.pth'
        if os.path.exists(embedding_path):
            target_se = load_embedding(embedding_path)
    else:
        st.info("You haven't uploaded any voices yet. Please upload an audio sample above.")

# Load source accent safely
def load_accent(accent_name):
    try:
        return torch.load(f'{ckpt_base}/{accent_name}.pth').to(device)
    except Exception as e:
        st.error(f"Error loading accent: {str(e)}")
        return None

source_se = load_accent(accent)

# Generate speech
if st.button("Generate Speech"):
    if target_se is None:
        st.error("Please select or upload a voice first.")
    elif source_se is None:
        st.error("Failed to load accent. Please try a different accent.")
    else:
        with st.spinner("Generating speech..."):
            try:
                # Run the base speaker TTS
                src_path = f'{output_dir}/tmp.wav'
                base_speaker_tts.tts(text_input, src_path, speaker=emotion, language='English', speed=speed)
                
                # Run the tone color converter
                output_path = f'{output_dir}/output_{accent}_{int(speed*100)}.wav'
                encode_message = "@OpenVoice"
                tone_color_converter.convert(
                    audio_src_path=src_path,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=output_path,
                    message=encode_message
                )
                
                # Display audio player
                st.subheader("Generated Speech")
                st.audio(output_path)
                
                # Add download button
                with open(output_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Speech",
                        data=file,
                        file_name=f"openvoice_generated_{accent}_{int(speed*100)}.wav",
                        mime="audio/wav"
                    )
            except Exception as e:
                st.error(f"Error generating speech: {str(e)}")

# Info
st.markdown("---")
st.markdown("""
### About Parrot TTS
Parrot TTS is a text-to-speech application that can generate speech using a target voice
This app uses OpenVoice, a versatile voice cloning system that can preserve 
expressive, singing, and stylistic characteristics of a target voice.

#### How to use:
1. Enter the text you want to convert to speech
2. Choose an accent and adjust speech settings
3. Select a preset voice(For Generic TTS) or upload and use your own voice sample(For Voice Cloning)(20-30 seconds)
4. Click "Generate Speech" to create the audio
""")