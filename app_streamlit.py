import streamlit as st
import tempfile
import os
import shutil
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile as wav
from pydub import AudioSegment
from inference import predict

st.set_page_config(page_title="Baby Cry Translator", layout="centered")
st.title("👶 Baby Cry Translator")
st.write("Upload OR record baby's cry to detect the reason.")

############################################################
# 🎤 RECORD AUDIO
############################################################
def record_audio(seconds=3, sr=16000):
    st.info(f"🎙 Recording for {seconds} seconds...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(tmp.name, sr, audio)
    tmp.close()
    return tmp.name

############################################################
# 🔊 TEXT TO SPEECH
############################################################
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

############################################################
# 🎧 CONVERT TO WAV (ONLY IF NOT WAV)
############################################################
def to_wav(src, dest):
    audio = AudioSegment.from_file(src)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(dest, format="wav")

############################################################
# 📌 FILE UPLOADER
############################################################
uploaded = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "ogg"])
record_btn = st.button("🎤 Record 3 sec Audio")

############################################################
# 🔍 PROCESS UPLOADED AUDIO
############################################################
if uploaded:

    suffix = uploaded.name.lower().split(".")[-1]
    raw = tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix)
    raw.write(uploaded.read())
    raw.close()

    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_file.close()

    if suffix == "wav":
        shutil.copy(raw.name, wav_file.name)
    else:
        to_wav(raw.name, wav_file.name)

    label, conf, _ = predict(wav_file.name)
    st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")

    if st.button("🔊 Speak Result"):
        speak(f"The baby is likely {label}")

    # SAFE DELETE AFTER MODEL USE
    for f in [raw.name, wav_file.name]:
        try:
            os.remove(f)
        except:
            pass

############################################################
# 🔍 PROCESS RECORDED AUDIO
############################################################
elif record_btn:

    rec_file = record_audio(3)

    label, conf, _ = predict(rec_file)
    st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")

    if st.button("🔊 Speak Recorded Result"):
        speak(f"The baby is likely {label}")

    try:
        os.remove(rec_file)
    except:
        pass

############################################################
st.markdown("---")
st.caption("Best accuracy: use 2–3 sec cry clips.")
