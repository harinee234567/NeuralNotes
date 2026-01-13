import streamlit as st
import numpy as np
import librosa
import pickle
import warnings
from tensorflow import keras
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import requests
from io import BytesIO

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎵",
    layout="wide"
)

# Load model and preprocessors
@st.cache_resource
def load_artifacts():
    try:
        model = keras.models.load_model('artifacts/emotion_model_final.keras')
        with open('artifacts/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('artifacts/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, label_encoder, scaler
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

# Feature extraction functions
def extract_features(data, sample_rate):
    # Extract audio features from audio data
    result = np.array([])
    
    # Calculate appropriate n_fft based on data length
    n_fft = min(2048, len(data))
    hop_length = n_fft // 4

    try:
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result = np.hstack((result, zcr))

        # Chroma STFT
        stft = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft))

        # MFCC (mean + std)
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        result = np.hstack((result, mfcc_mean, mfcc_std))

        # RMS Energy
        rms = np.mean(librosa.feature.rms(y=data, frame_length=n_fft, hop_length=hop_length).T, axis=0)
        result = np.hstack((result, rms))

        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length).T, axis=0)
        result = np.hstack((result, mel))

        # Spectral features
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length).T, axis=0)
        result = np.hstack((result, rolloff))

        centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length).T, axis=0)
        result = np.hstack((result, centroid))

        contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length).T, axis=0)
        result = np.hstack((result, contrast))

        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length).T, axis=0)
        result = np.hstack((result, bandwidth))

        # Tonnetz
        try:
            tonnetz = np.mean(librosa.feature.tonnetz(y=data, sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
        except:
            tonnetz = np.zeros(6)
            result = np.hstack((result, tonnetz))

        return result
    
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return np.array([])

def record_audio(duration=3, sample_rate=22050):
    """Record audio from microphone"""
    try:
        st.info(f"🎤 Recording for {duration} seconds... Speak now!")
        recording = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        st.success("✅ Recording complete!")
        return recording.flatten(), sample_rate
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None, None

def process_audio_data(audio_data, sr):
    """Process audio data and extract features"""
    try:
        # Use a segment from the middle
        start_sample = int(0.6 * sr)
        end_sample = int(3.1 * sr)
        
        if len(audio_data) > end_sample:
            data = audio_data[start_sample:end_sample]
        else:
            data = audio_data
        
        # Ensure minimum length
        if len(data) < 512:
            data = np.pad(data, (0, 512 - len(data)), mode='constant')
        
        # Extract features
        features = extract_features(data, sr)
        
        return features
    
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Song recommendations based on emotion and artist
# Load song recommendations from text file
def load_song_recommendations():
    """Load song recommendations from external text file"""
    file_path = 'songs_data.txt'
    recommendations = {}
    current_emotion = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                if line.endswith(':') and line.count(':') == 1:
                    current_emotion = line[:-1].strip().lower()
                    recommendations[current_emotion] = {}
                
                elif ':' in line and current_emotion:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        artist = parts[0].strip()
                        songs = [song.strip() for song in parts[1].split(',')]
                        recommendations[current_emotion][artist] = songs
        
        return recommendations
    
    except FileNotFoundError:
        st.error(f"Song file not found: {file_path}")
        return {}
    except Exception as e:
        st.error(f"Error loading songs: {e}")
        return {}

# Load the song recommendations at module level
SONG_RECOMMENDATIONS = load_song_recommendations()

# Emotion color mapping
EMOTION_COLORS = {
    'angry': '#FF4444',
    'happy': '#FFD700',
    'sad': '#4169E1',
    'neutral': '#808080',
    'fear': '#800080',
    'disgust': '#8B4513',
    'surprise': '#FF69B4'
}

# Main app
def main():
    st.title("🎵 Speech Emotion Recognition & Music Recommender")
    st.markdown("Record your voice to detect emotion and get personalized song recommendations!")
    
    # Load model artifacts
    model, label_encoder, scaler = load_artifacts()
    
    if model is None:
        st.error("Failed to load model. Please ensure all artifacts are in the 'artifacts' folder.")
        return
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    
    artist = st.sidebar.selectbox(
        "🎤 Select Your Favorite Artist:",
       ['Taylor Swift', 'Ed Sheeran', 'Ariana Grande', 'Adele', 'The Weeknd', 'BTS', 'D.O.', 'Arctic Monkeys', 'Vampire Weekend', 'Phoebe Bridgers', 'Lana Del Rey', 'Radiohead', 'The Neighbourhood', 'Tame Impala']

    )
    
    duration = st.sidebar.slider(
        "⏱️ Recording Duration (seconds):",
        min_value=2,
        max_value=10,
        value=3
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Speak clearly and express your emotion for better detection!")
    
    # Initialize session state
    if 'recorded_audio' not in st.session_state:
        st.session_state.recorded_audio = None
    if 'sample_rate' not in st.session_state:
        st.session_state.sample_rate = None
    if 'emotion_detected' not in st.session_state:
        st.session_state.emotion_detected = None
    if 'temp_audio_path' not in st.session_state:
        st.session_state.temp_audio_path = None
    
    # Main content
    st.header("🎙️ Record Your Voice")
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if st.button("🔴 Start Recording", type="primary", use_container_width=True):
            audio_data, sr = record_audio(duration=duration)
            if audio_data is not None:
                st.session_state.recorded_audio = audio_data
                st.session_state.sample_rate = sr
                # Clear previous temp file if exists
                if st.session_state.temp_audio_path and os.path.exists(st.session_state.temp_audio_path):
                    try:
                        os.unlink(st.session_state.temp_audio_path)
                    except:
                        pass
                st.session_state.temp_audio_path = None
    
    with col2:
        if st.button("🗑️ Clear Recording", use_container_width=True):
            # Clean up temp file before clearing
            if st.session_state.temp_audio_path:
                try:
                    if os.path.exists(st.session_state.temp_audio_path):
                        os.unlink(st.session_state.temp_audio_path)
                except PermissionError:
                    pass  # File will be cleaned up eventually
            
            st.session_state.recorded_audio = None
            st.session_state.sample_rate = None
            st.session_state.emotion_detected = None
            st.session_state.temp_audio_path = None
            st.rerun()
    
    # Display recorded audio
    if st.session_state.recorded_audio is not None:
        st.success("✅ Audio recorded successfully!")
        
        # Save to temporary file for playback (only once)
        if st.session_state.temp_audio_path is None or not os.path.exists(st.session_state.temp_audio_path):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                sf.write(tmp_file.name, st.session_state.recorded_audio, st.session_state.sample_rate)
                st.session_state.temp_audio_path = tmp_file.name
        
        # Display audio player
        st.audio(st.session_state.temp_audio_path)
        
        # Analyze button
        if st.button("🔍 Analyze Emotion", type="secondary", use_container_width=True):
            with st.spinner("Processing audio and extracting features..."):
                # Extract features
                features = process_audio_data(st.session_state.recorded_audio, st.session_state.sample_rate)
                
                if features is not None and len(features) > 0:
                    # Scale features
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    
                    # Predict
                    prediction = model.predict(features_scaled, verbose=0)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    emotion = label_encoder.inverse_transform([predicted_class])[0]
                    confidence = prediction[0][predicted_class] * 100
                    
                    st.session_state.emotion_detected = {
                        'emotion': emotion,
                        'confidence': confidence,
                        'prediction': prediction
                    }
                else:
                    st.error("Failed to extract features from the audio.")
    
    # Display results
    if st.session_state.emotion_detected is not None:
        st.markdown("---")
        st.success("✅ Analysis Complete!")
        
        emotion = st.session_state.emotion_detected['emotion']
        confidence = st.session_state.emotion_detected['confidence']
        prediction = st.session_state.emotion_detected['prediction']
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎭 Detected Emotion")
            st.markdown(f"""
            <div style='background-color: {EMOTION_COLORS.get(emotion, '#808080')}; 
                        padding: 20px; 
                        border-radius: 10px; 
                        text-align: center;'>
                <h1 style='color: white; margin: 0;'>{emotion.upper()}</h1>
                <p style='color: white; font-size: 18px; margin: 10px 0 0 0;'>
                    Confidence: {confidence:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show all probabilities
            st.subheader("📊 All Emotion Probabilities")
            prob_df = []
            for i, prob in enumerate(prediction[0]):
                emotion_name = label_encoder.inverse_transform([i])[0]
                prob_df.append({
                    'Emotion': emotion_name.capitalize(),
                    'Probability': f"{prob * 100:.2f}%"
                })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader(f"🎵 Song Recommendations")
            
            # Get recommendations
            recommendations = SONG_RECOMMENDATIONS.get(emotion, {}).get(artist, [])
            
            if recommendations:
                st.markdown(f"**Based on your {emotion} emotion, here are songs by {artist}:**")
                
                # Display songs with search option
                for i, song in enumerate(recommendations[:3], 1):
                    st.markdown(f"### {i}. {song}")
                    
                    # Search query
                    search_query = f"{artist} {song}"
                    search_url = f"https://soundcloud.com/search?q={search_query.replace(' ', '+')}"
                    
                    st.markdown(f"""
                    <a href="{search_url}" target="_blank" style="text-decoration: none;">
                        <button style="background-color: #ff5500; color: white; padding: 10px 20px; 
                                       border: none; border-radius: 5px; cursor: pointer; 
                                       font-size: 14px; margin: 5px 0;">
                            🎧 Play on SoundCloud
                        </button>
                    </a>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
            else:
                st.info(f"No specific recommendations available for {emotion} emotion and {artist}.")
            
            # Additional info
            st.markdown("**💡 Why these songs?**")
            emotion_descriptions = {
                'angry': "These energetic tracks match your intense mood!",
                'happy': "Uplifting songs to maintain your positive vibes!",
                'sad': "Soothing melodies to comfort and understand you.",
                'neutral': "Balanced tracks for your calm state.",
                'fear': "Empowering songs to help you feel stronger.",
                'disgust': "Bold tracks that match your assertive mood.",
                'surprise': "Dynamic songs for your spontaneous feeling!"
            }
            st.write(emotion_descriptions.get(emotion, "Songs selected based on your emotion."))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit | Speech Emotion Recognition</p>
        <p style='font-size: 12px;'>Note: Make sure your microphone is enabled and working</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()