import streamlit as st
import tempfile
import os
import shutil
from pydub import AudioSegment
from inference import predict

# NEW: Browser-based audio recorder (no sounddevice needed!)
from st_audiorec import st_audiorec

st.set_page_config(page_title="Baby Cry Translator", layout="centered")
st.title("Baby Cry Translator")
st.write("Upload OR record baby's cry to detect the reason.")

# ------------------------------------------------------------------
# CLIENT-SIDE AUDIO RECORDING (works in browser, no PortAudio!)
# ------------------------------------------------------------------
st.markdown("### Record 3-second cry")
wav_audio_data = st_audiorec()

if wav_audio_data:
    # Save recorded audio to temp file
    rec_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    rec_file.write(wav_audio_data)
    rec_file.close()

    label, conf, _ = predict(rec_file.name)
    st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")

    # Browser-based speech (no pyttsx3!)
    st.audio(wav_audio_data, format="audio/wav")
    st.markdown(
        f'<script>const msg = new SpeechSynthesisUtterance("The baby is likely {label}"); '
        f'window.speechSynthesis.speak(msg);</script>',
        unsafe_allow_html=True
    )

    # Cleanup
    os.unlink(rec_file.name)

# ------------------------------------------------------------------
# FILE UPLOADER (unchanged, works perfectly)
# ------------------------------------------------------------------
st.markdown("### Or upload an audio file")
uploaded = st.file_uploader("Choose audio file", type=["wav", "mp3", "m4a", "ogg", "mp4"])

if uploaded:
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
    st.audio(wav_file.name)

    # Browser TTS instead of pyttsx3
    if st.button("Speak Result"):
        st.markdown(
            f'<script>const msg = new SpeechSynthesisUtterance("The baby is likely {label}"); '
            f'window.speechSynthesis.speak(msg);</script>',
            unsafe_allow_html=True
        )

    # Cleanup
    for f in [raw.name, wav_file.name]:
        try:
            os.remove(f)
        except:
            pass

st.markdown("---")
st.caption("Best accuracy with 2–5 second clear cry clips.")
