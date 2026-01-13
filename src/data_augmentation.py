import numpy as np
import pandas as pd
import librosa
import random
from tqdm import tqdm

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data.astype(np.float32), rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch_shift(data, sampling_rate, pitch_factor=2):
    return librosa.effects.pitch_shift(data.astype(np.float32), n_steps=pitch_factor, sr=sampling_rate)


def apply_limited_augmentation(train_csv_path, output_csv_path, target_total=14000):
    """
    Apply LIMITED augmentation to training set only.
    Goal: Increase from ~9,725 to ~14,000 samples (40% increase)
    """
    print("\n" + "="*50)
    print("STEP 3: LIMITED AUGMENTATION (TRAIN SET ONLY)")
    print("="*50)

    train_df = pd.read_csv(train_csv_path)
    train_paths = train_df['AudioPath'].values
    train_labels = train_df['Label'].values

    original_count = len(train_df)

    print(f"\nOriginal training samples: {original_count}")

    # Calculate augmented samples needed
    num_augmented_needed = target_total - original_count
    print(f"Augmented samples to create: {num_augmented_needed}")

    aug_records = []

    # Step 1: Add ALL original samples
    print("\nAdding original samples...")
    for path, label in zip(train_paths, train_labels):
        aug_records.append({
            'AudioPath': path,
            'Label': label,
            'Augmentation': 'original'
        })

    # Step 2: Randomly select samples to augment
    print(f"Selecting {num_augmented_needed} samples for augmentation...")
    samples_to_augment = []

    # If we need more augmented samples than we have originals, cycle through
    num_cycles = (num_augmented_needed // original_count) + 1
    for _ in range(num_cycles):
        shuffled_indices = np.random.permutation(original_count)
        for idx in shuffled_indices:
            samples_to_augment.append({
                'path': train_paths[idx],
                'label': train_labels[idx]
            })

    # Trim to exact number
    samples_to_augment = samples_to_augment[:num_augmented_needed]

    # Step 3: Add augmented samples (with random augmentation type)
    print("Applying augmentations...")
    for record in tqdm(samples_to_augment):
        aug_type = random.choice(['noise', 'stretch', 'pitch', 'shift'])
        aug_records.append({
            'AudioPath': record['path'],
            'Label': record['label'],
            'Augmentation': aug_type
        })

    # Create DataFrame
    train_aug_df = pd.DataFrame(aug_records)
    train_aug_df.to_csv(output_csv_path, index=False)

    print(f"\n{'='*50}")
    print("Augmentation Summary:")
    print(f"{'='*50}")
    print(f"Original samples: {original_count}")
    print(f"Augmented samples: {len(train_aug_df) - original_count}")
    print(f"Total samples: {len(train_aug_df)}")
    print(f"Augmentation factor: {len(train_aug_df)/original_count:.2f}x")

    print("\n\nAugmentation type distribution:")
    print(train_aug_df['Augmentation'].value_counts())

    print(f"\n\nSaved to: {output_csv_path}")
    print("="*50 + "\n")

    return train_aug_df

if __name__ == "__main__":
    apply_limited_augmentation(
        "Split/train.csv",
        "Split/train_augmented.csv",
        target_total=14000
    )