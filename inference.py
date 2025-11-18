# inference.py
import numpy as np
import librosa
import json
import tensorflow as tf

# --- MUST BE IDENTICAL TO TRAINING ---
SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024


# -------------------------------------------------------
# Load model + labels
# -------------------------------------------------------
def load_model(path="models/baby_cry_model_final.keras"):
    model = tf.keras.models.load_model(path)
    with open("models/labels.json") as f:
        labels = json.load(f)
    return model, labels


# -------------------------------------------------------
# Preprocess audio to mel-spectrogram (IDENTICAL to train)
# -------------------------------------------------------
def preprocess_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    # Pad or trim to fixed duration (3 sec)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-9)

    # Expand dims → (1, 64, 94, 1)
    mel_db = mel_db[..., np.newaxis]
    mel_db = np.expand_dims(mel_db, axis=0)

    return mel_db


# -------------------------------------------------------
# Prediction function
# -------------------------------------------------------
def predict(audio_path):
    model, labels = load_model()
    mel = preprocess_audio(audio_path)

    preds = model.predict(mel)[0]

    idx = np.argmax(preds)
    confidence = float(preds[idx])

    return labels[idx], confidence, preds.tolist()


# -------------------------------------------------------
# Run from command-line
# -------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py path/to/file.wav")
        exit()

    audio_file = sys.argv[1]
    label, conf, probs = predict(audio_file)

    print("\n🎧 Baby Cry Prediction")
    print("-------------------------")
    print(f"Predicted: {label.upper()} ({conf*100:.2f}% confidence)")
