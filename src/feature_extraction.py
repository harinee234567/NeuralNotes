import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal of length=.*')


def noise(data):
    """Add random noise to audio data."""
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    """Time stretch audio data."""
    # Add n_fft parameter based on data length
    n_fft = min(2048, len(data))
    return librosa.effects.time_stretch(data, rate=rate, n_fft=n_fft)


def pitch_shift(data, sample_rate, n_steps=0):
    """Pitch shift audio data."""
    if n_steps == 0:
        n_steps = np.random.randint(-5, 5)
    
    # Add n_fft parameter based on data length
    n_fft = min(2048, len(data))
    return librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=n_steps, n_fft=n_fft)


def shift(data):
    """Shift audio data along the time axis."""
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def extract_features(data, sample_rate):
    """
    Extract aggregated audio features.
    Returns ONE feature vector per audio file.
    """
    result = np.array([])
    
    # Calculate appropriate n_fft based on data length
    n_fft = min(2048, len(data))
    hop_length = n_fft // 4

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
    except Exception:
        # If tonnetz fails, add zeros instead
        tonnetz = np.zeros(6)  # tonnetz returns 6 features
        result = np.hstack((result, tonnetz))
    return result


def get_features_with_augmentation(path, aug_type):
    """Load audio, apply augmentation if needed, extract features."""
    try:
        data, sr = librosa.load(path, duration=2.5, offset=0.6)

        if len(data) < 512:
            data = np.pad(data, (0, 512 - len(data)), mode='constant')

        # Apply augmentation
        if aug_type == 'noise':
            data = noise(data)
        elif aug_type == 'stretch':
            data = stretch(data)
        elif aug_type == 'pitch':
            data = pitch_shift(data, sr)
        elif aug_type == 'shift':
            data = shift(data)

        # Extract features
        features = extract_features(data, sr)
        return features

    except Exception as e:
        print(f"Error: {path} - {e}")
        return np.array([])


def extract_features_from_csv(csv_path, output_path, dataset_name):
    """Extract features from CSV (with or without augmentation column)."""
    print(f"\n{'='*50}")
    print(f"EXTRACTING FEATURES: {dataset_name}")
    print(f"{'='*50}")

    df = pd.read_csv(csv_path)

    # Add 'Augmentation' column if it doesn't exist (for val/test)
    if 'Augmentation' not in df.columns:
        df['Augmentation'] = 'original'

    print(f"Processing {len(df)} samples...")

    features_list = []
    labels_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        features = get_features_with_augmentation(row['AudioPath'], row['Augmentation'])

        if len(features) > 0:
            features_list.append(features)
            labels_list.append(row['Label'])

    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['Label'] = labels_list

    # Save
    features_df.to_csv(output_path, index=False)

    print(f"\nExtracted {len(features_df)} feature vectors")
    print(f"Feature dimensions: {features_df.shape[1] - 1}")  # -1 for Label column
    print(f"Saved to: {output_path}")
    print(f"{'='*50}\n")

    return features_df


if __name__ == "__main__":
    # Example usage
    # You can modify these paths as needed
    
    # Extract features from training data
    train_features = extract_features_from_csv(
        csv_path='Split/train_augmented.csv',
        output_path='Split/Features/train_features.csv',
        dataset_name='Training Set'
    )
    
    # Extract features from validation data
    val_features = extract_features_from_csv(
        csv_path='Split/val.csv',
        output_path='Split/Features/val_features.csv',
        dataset_name='Validation Set'
    )
    
    # Extract features from test data
    test_features = extract_features_from_csv(
        csv_path='Split/test.csv',
        output_path='Split/Features/test_features.csv',
        dataset_name='Test Set'
    )