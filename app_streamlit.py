import streamlit as st
import tempfile
import os
import shutil
from pydub import AudioSegment
from inference import predict

# Fixed import: Matches the PyPI package name
from streamlit_audiorec import st_audiorec  # Note: underscore in import

st.set_page_config(page_title="Baby Cry Translator", layout="centered")
st.title("👶 Baby Cry Translator")
st.write("Upload OR record baby's cry to detect the reason.")

############################################################
# 🎤 RECORD AUDIO (Browser-based, no PortAudio!)
############################################################
st.markdown("### 🎙 Record 3-Second Cry")
wav_audio_data = st_audiorec()  # Records in browser, returns WAV bytes

if wav_audio_data is not None:
    # Save to temp file for prediction
    rec_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    rec_file.write(wav_audio_data)
    rec_file.close()

    # Predict
    label, conf, _ = predict(rec_file.name)
    st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")

    # Preview the recording
    st.audio(wav_audio_data, format="audio/wav")

    # Browser TTS (no pyttsx3 needed!)
    if st.button("🔊 Speak Result"):
        st.markdown(
            f'<script>const msg = new SpeechSynthesisUtterance("The baby is likely {label}"); '
            f'msg.lang = "en-US"; window.speechSynthesis.speak(msg);</script>',
            unsafe_allow_html=True
        )

    # Cleanup
    try:
        os.remove(rec_file.name)
    except:
        pass

############################################################
# 📎 FILE UPLOADER (Unchanged, works great)
############################################################
st.markdown("### 📁 Or Upload Audio")
uploaded = st.file_uploader("Choose audio file", type=["wav", "mp3", "m4a", "ogg", "mp4"])

if uploaded is not None:
    suffix = uploaded.name.lower().split(".")[-1]
    raw = tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix)
    raw.write(uploaded.getvalue())
    raw.close()

    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_file.close()

    if suffix != "wav":
        audio = AudioSegment.from_file(raw.name)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_file.name, format="wav")
    else:
        shutil.copy(raw.name, wav_file.name)

    # Predict
    label, conf, _ = predict(wav_file.name)
    st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")

    # Preview
    st.audio(wav_file.name, format="audio/wav")

    # Browser TTS
    if st.button("🔊 Speak Result"):
        st.markdown(
            f'<script>const msg = new SpeechSynthesisUtterance("The baby is likely {label}"); '
            f'msg.lang = "en-US"; window.speechSynthesis.speak(msg);</script>',
            unsafe_allow_html=True
        )

    # Cleanup
    for f in [raw.name, wav_file.name]:
        try:
            os.remove(f)
        except:
            pass

############################################################
st.markdown("---")
st.caption("Best accuracy: Use 2–3 sec clear cry clips. Works on mobile too!")
