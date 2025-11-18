import streamlit as st
import tempfile
import os
import shutil
from pydub import AudioSegment
from inference import predict

# CORRECT import for the streamlit-audiorec component
from st_audiorec import st_audiorec

st.set_page_config(page_title="Baby Cry  Translator", layout="centered")
st.title("Baby Cry Translator")
st.write("Upload OR record baby's cry to detect the reason.")

# ────────────────────────────── RECORD AUDIO ──────────────────────────────
st.markdown("### Record 3-Second Cry")
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    rec_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    rec_file.write(wav_audio_data)
    rec_file.close()

    label, conf, _ = predict(rec_file.name)
    st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")
    st.audio(wav_audio_data, format="audio/wav")

    if st.button("Speak Result (Recording)", key="speak_rec"):
        st.markdown(
            f'<script>const msg = new SpeechSynthesisUtterance("The baby is likely {label}"); '
            f'msg.lang = "en-US"; window.speechSynthesis.speak(msg);</script>',
            unsafe_allow_html=True
        )

    os.unlink(rec_file.name)

# ────────────────────────────── FILE UPLOADER ──────────────────────────────
st.markdown("### Or Upload Audio File")
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

    label, conf, _ = predict(wav_file.name)
    st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")
    st.audio(wav_file.name, format="audio/wav")

    if st.button("Speak Result (Upload)", key="speak_up"):
        st.markdown(
            f'<script>const msg = new SpeechSynthesisUtterance("The baby is likely {label}"); '
            f'msg.lang = "en-US"; window.speechSynthesis.speak(msg);</script>',
            unsafe_allow_html=True
        )

    for f in [raw.name, wav_file.name]:
        try:
            os.remove(f)
        except:
            pass

# ────────────────────────────── FOOTER ──────────────────────────────
st.markdown("---")
st.caption("Best accuracy: Use 2–3 sec clear cry clips. Works on mobile too!")
