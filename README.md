# NeuralNotes - Speech Emotion Recognition Music App

## Overview
The **Speech Emotion Recognition Music App** is an intelligent application that detects human emotions from speech input and plays suitable music using **SoundCloud** based on the recognized emotion.  
The goal of this project is to enhance user experience by combining **speech processing**, **machine learning**, and **emotion-aware music recommendation**.

---

## Features
- Speech input through microphone or audio files  
- Emotion detection from speech signals  
- Emotion classification (Happy, Sad, Angry, Neutral, Fearful, Disgust, Surprise)  
- Automatic music playback from SoundCloud based on detected emotion  
- User-friendly and interactive interface  

---

## Emotion Categories & Music Mapping
| Emotion   | Music Type Played |
|----------|------------------|
| Happy    | Upbeat, energetic tracks |
| Sad      | Calm, soothing tracks |
| Angry    | Relaxing, soft tracks |
| Neutral  | Light, ambient music |
| Fear     | Comforting, slow music |
| Disgust  | Clean, mellow, mood-lifting tracks |
| Surprise | Energetic, exciting, dynamic tracks |

---

## Technologies Used
- **Python**
- **Machine Learning / Deep Learning**
- **Librosa** – audio feature extraction  
- **NumPy, Pandas** – data processing  
- **Scikit-learn / TensorFlow / PyTorch** – model training  
- **SoundCloud** – music streaming  
- **Streamlit** – web interface  

---

## System Architecture
1. Capture speech input  
2. Preprocess audio signal  
3. Extract speech features (MFCC, Chroma, Spectral features, etc.)  
4. Predict emotion using trained ML model  
5. Play emotion-based music in SoundCloud  

---

## Live Preview
- https://neural-notes.streamlit.app/

---

### How to Use the App
1. **Select your favourite artist** from the drop-down and **record duration (in seconds)** from the right pane.  
2. Click on **"Click to Record"** and record your audio.  
   - ⚠️ **Note:** Make sure your microphone is enabled and working.  
   - While recording, the **microphone emoji is red**.  
   - After recording is complete, the **microphone emoji turns blue**.  
3. Click on **"Analyze Emotion"** to detect your emotions and get song recommendations.  
4. Pick a song from the recommended list and click on **"Play on SoundCloud"** to listen.  

---

## Developers
- **Harineesha Nutakki**  
- **Attili Valli Sai Meghana**
