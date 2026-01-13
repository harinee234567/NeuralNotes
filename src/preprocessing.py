import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "AudioFiles")

# RAVDESS DATASET
def ravdess_data():
    ravdess_path = os.path.join(AUDIO_DIR, "RAVDESS")

    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }

    data = []

    for file in os.listdir(ravdess_path):
        if not file.lower().endswith(".wav"):
            continue

        parts = file.split("-")
        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        if emotion_code in emotion_map:
            data.append({
                "AudioPath": os.path.join(ravdess_path, file),
                "Label": emotion_map[emotion_code]
            })

    df = pd.DataFrame(data)
    print("RAVDESS files:", len(df))
    return df


# CREMA-D DATASET
def crema_data():
    crema_path = os.path.join(AUDIO_DIR, "CREMA-D")

    emotion_map = {
        "SAD": "sad",
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral"
    }

    data = []

    for file in os.listdir(crema_path):
        if not file.lower().endswith(".wav"):
            continue

        parts = file.split("_")
        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        if emotion_code in emotion_map:
            data.append({
                "AudioPath": os.path.join(crema_path, file),
                "Label": emotion_map[emotion_code]
            })

    df = pd.DataFrame(data)
    print("CREMA-D files:", len(df))
    return df


# TESS DATASET
def tess_data():
    tess_path = os.path.join(AUDIO_DIR, "TESS")
    data = []

    for file in os.listdir(tess_path):
        if not file.lower().endswith(".wav"):
            continue

        parts = file.split("_")
        if len(parts) < 2:
            continue

        emotion = parts[-1].replace(".wav", "").lower()

        data.append({
            "AudioPath": os.path.join(tess_path, file),
            "Label": emotion
        })

    df = pd.DataFrame(data)
    print("TESS files:", len(df))
    return df



# SAVEE DATASET
def savee_data():
    savee_path = os.path.join(AUDIO_DIR, "ALL")

    emotion_map = {
        "a": "angry",
        "d": "disgust",
        "f": "fear",
        "h": "happy",
        "n": "neutral",
        "sa": "sad",
        "su": "surprise"
    }

    data = []

    for file in os.listdir(savee_path):
        if not file.lower().endswith(".wav"):
            continue

        parts = file.split("_")
        if len(parts) < 2:
            continue

        match = re.match(r"([a-z]+)(\d+)", parts[1].lower())
        if not match:
            continue

        emotion_code = match.group(1)
        if emotion_code in emotion_map:
            data.append({
                "AudioPath": os.path.join(savee_path, file),
                "Label": emotion_map[emotion_code]
            })

    df = pd.DataFrame(data)
    print("SAVEE files:", len(df))
    return df



# FETCH & COMBINE DATASETS
def fetch_data():
    df_ravdess = ravdess_data()
    df_crema = crema_data()
    df_tess = tess_data()
    df_savee = savee_data()

    print("\nUnique labels per dataset:")
    print("RAVDESS:", df_ravdess.Label.unique())
    print("CREMA-D:", df_crema.Label.unique())
    print("TESS:", df_tess.Label.unique())
    print("SAVEE:", df_savee.Label.unique())

    df_combined = pd.concat(
        [df_ravdess, df_crema, df_tess, df_savee],
        ignore_index=True
    )

    print("\nCombined labels:", df_combined.Label.unique())
    print("Total files:", len(df_combined))

    save_path = os.path.join(BASE_DIR, "preprocesseddata.csv")
    df_combined.to_csv(save_path, index=False)

    print("Saved CSV to:", save_path)
    return df_combined



# ADDITIONAL PREPROCESSING (CLEAN & SAFE)

def additional_preprocess(filepath):
    df = pd.read_csv(filepath)

    print("\n" + "=" * 50)
    print("STEP 1: ADDITIONAL PREPROCESSING")
    print("=" * 50)

    print(f"\nOriginal dataset size: {len(df)}")
    print("Original labels:", df["Label"].unique())

    # Normalize labels
    df["Label"] = df["Label"].str.lower().str.strip()

    # Explicit label standardization
    label_map = {
        "sadness": "sad",
        "sad": "sad",
        "happiness": "happy",
        "happy": "happy",
        "fearful": "fear",
        "fear": "fear",
        "anger": "angry",
        "angry": "angry",
        "pleasant_surprise": "surprise",
        "pleasant_surprised": "surprise",
        "surprised": "surprise",
        "ps": "surprise",
        "neutral": "neutral",
        "calm": "calm",
        "disgust": "disgust"
    }

    df["Label"] = df["Label"].map(label_map).fillna(df["Label"])

    print("\nLabel counts AFTER standardization:")
    print(df["Label"].value_counts(dropna=False))

    # Drop underrepresented emotions
    drop_labels = ["surprise", "calm"]
    df = df[~df["Label"].isin(drop_labels)]

    print("\nDropped labels:", drop_labels)
    print("Label counts AFTER dropping:")
    print(df["Label"].value_counts())

    print(f"\nFinal dataset size: {len(df)}")

    print("=" * 50 + "\n")

    return df

# SPLIT DATASET

def split_dataset(df, output_dir):
    # Split into Train (80%), Validation (10%), Test (10%)
    df_train, df_temp = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        shuffle=True,
        stratify=df['Label']
    )

    # Second split: Split temp into 50% val, 50% test (each 10% of original)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=42,
        shuffle=True,
        stratify=df_temp['Label']
    )

    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    df_train.to_csv(f"{output_dir}/train.csv", index=False)
    df_val.to_csv(f"{output_dir}/val.csv", index=False)
    df_test.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"\nTrain set: {len(df_train)} samples")
    print(f"Validation set: {len(df_val)} samples")
    print(f"Test set: {len(df_test)} samples")

    print("\n\nTrain set label distribution:")
    print(df_train['Label'].value_counts())

    print(f"\n\nFiles saved to: {output_dir}/")
    print("="*50 + "\n")

    return df_train, df_val, df_test


if __name__ == "__main__":
    fetch_data()

    df_clean = additional_preprocess(
        os.path.join(BASE_DIR, "preprocesseddata.csv")
    )
    split_dataset(df_clean, os.path.join(BASE_DIR, "Split"))