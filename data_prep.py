# data_prep_mel.py
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

# CONFIG
DATA_DIR = "data"         # expected: data/<label>/*.wav
SAMPLE_RATE = 16000
DURATION = 3.0            # seconds (trim/pad)
N_MELS = 64               # number of mel bands
HOP_LENGTH = 512
N_FFT = 1024

def load_audio(path, sr=SAMPLE_RATE, duration=DURATION):
    """Load audio and pad or truncate to fixed duration."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    if len(y) < int(sr * duration):
        pad_len = int(sr * duration) - len(y)
        y = np.pad(y, (0, pad_len))
    return y

def augment_audio(y, sr):
    """Create multiple augmented variations of the same audio (Librosa 0.8 - 0.10+ compatible)."""
    import librosa
    augmented = [y]  # include original always

    try:
        # Pitch shift
        if hasattr(librosa.effects, "pitch_shift"):
            augmented.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2))
            augmented.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2))
        elif hasattr(librosa.effects, "piecewise_pitch_shift"):
            augmented.append(librosa.effects.piecewise_pitch_shift(y=y, sr=sr, steps=2, bins_per_octave=12))
            augmented.append(librosa.effects.piecewise_pitch_shift(y=y, sr=sr, steps=-2, bins_per_octave=12))
    except Exception as e:
        print(f"⚠️ Pitch shift skipped: {e}")

    try:
        # Time stretch
        augmented.append(librosa.effects.time_stretch(y, rate=1.1))
        augmented.append(librosa.effects.time_stretch(y, rate=0.9))
    except Exception as e:
        print(f"⚠️ Time stretch skipped: {e}")

    # Add background noise augmentation
    try:
        noise = np.random.randn(len(y))
        y_noisy = y + 0.005 * noise
        augmented.append(y_noisy)
    except Exception as e:
        print(f"⚠️ Noise augmentation skipped: {e}")

    # Remove invalid augmentations
    augmented = [a for a in augmented if a is not None and len(a) > 0]
    return augmented

def extract_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS):
    """Extract normalized log-mel spectrogram features."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # convert to decibels
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-9)  # normalization
    return mel_db

def prepare_dataset(data_dir=DATA_DIR, out_json="metadata_mel.json"):
    X, y = [], []
    labels = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    print("🎵 Found labels:", labels)

    for lab in labels:
        folder = Path(data_dir) / lab
        for wav in folder.glob("*.wav"):
            try:
                audio = load_audio(str(wav))
                # Apply augmentation (6× per file including original)
                for y_aug in augment_audio(audio, SAMPLE_RATE):
                    mel = extract_mel(y_aug)
                    fixed_frames = int(np.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH))
                    if mel.shape[1] < fixed_frames:
                        mel = np.pad(mel, ((0, 0), (0, fixed_frames - mel.shape[1])), mode='constant')
                    elif mel.shape[1] > fixed_frames:
                        mel = mel[:, :fixed_frames]
                    X.append(mel)
                    y.append(label_to_idx[lab])
            except Exception as e:
                print(f"❌ Failed {wav}: {e}")

    X = np.array(X)
    y = np.array(y)
    X = X[..., np.newaxis]  # (N, n_mels, frames, 1)
    print("\n✅ Log-Mel dataset created successfully!")
    print(f"📦 X shape: {X.shape} | y shape: {y.shape}")

    # Split dataset: train/val/test = 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Save metadata
    meta = {
        "labels": labels,
        "X_train_shape": X_train.shape,
        "X_val_shape": X_val.shape,
        "X_test_shape": X_test.shape
    }
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)

    # Save dataset arrays
    os.makedirs("models", exist_ok=True)
    np.save("models/X_train_mel.npy", X_train)
    np.save("models/X_val_mel.npy", X_val)
    np.save("models/X_test_mel.npy", X_test)
    np.save("models/y_train_mel.npy", y_train)
    np.save("models/y_val_mel.npy", y_val)
    np.save("models/y_test_mel.npy", y_test)

    print("\n✅ Data successfully saved to 'models/' directory")
    print("🗂️ Metadata file:", out_json)

if __name__ == "__main__":
    prepare_dataset()
