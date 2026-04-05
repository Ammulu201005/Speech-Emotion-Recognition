"""
Streamlit main app (Chat-first).

Run:
python -m streamlit run app/app.py

What you get:
- Chatbot opens first + bot greets first
- Chat supports text + mic voice + upload audio + upload video (audio extracted)
- SER model used for audio-based emotion detection (best_model.h5)
- Sidebar shows emotion, intensity, timeline graph, suggestions, songs (songs only once)
- Analyzer tab is separate (optional)
- Voice-to-voice support:
    * user voice -> Whisper STT
    * bot reply -> gTTS browser audio + optional pyttsx3 offline voice
"""

import os, sys, tempfile
import yt_dlp
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
from faster_whisper import WhisperModel

from utils.feature_extraction import extract_features
from utils.timeline import analyze_timeline, plot_timeline, get_timeline_summary
from utils.suggestions import get_suggestions
from models.intensity import TemperatureScaler

from app.chatbot import (
    chat,
    check_ollama_available,
    get_song_recos,
    maybe_speak_text,
    gemini_available,
    tts_to_mp3_bytes,
    get_detected_text_emotion,
    reset_session,
)

# ───────────────── CONFIG ─────────────────
st.set_page_config(
    page_title="EmotionGPT (Chat-first SER)",
    page_icon="🎙️",
    layout="wide",
)

LABELS = ['Neutral','Calm','Happy','Sad','Angry','Fearful','Disgust','Surprised']
EMOTION_EMOJI = {
    'Neutral':'😐', 'Calm':'😌', 'Happy':'😊', 'Sad':'😢',
    'Angry':'😠', 'Fearful':'😨', 'Disgust':'🤢', 'Surprised':'😲'
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved", "best_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "saved", "temperature_scaler.pkl")
SR = 22050

# IMPORTANT: Your model expects (128, 40) => MFCC ONLY (40)
FEATURE_TYPE_FIXED = "mfcc"

# ───────────────── CSS ─────────────────
st.markdown("""
<style>
.stApp{
  background:linear-gradient(135deg,#0f0f23 0%,#1a1a3e 100%);
}

.chat-user{
  background:rgba(99,102,241,0.22);
  border-radius:14px;
  padding:10px 12px;
  margin:6px 0;
  text-align:right;
}

.chat-bot{
  background:rgba(255,255,255,0.06);
  border-radius:14px;
  padding:10px 12px;
  margin:6px 0;
}

.badge{
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  font-size:0.85rem;
  border:1px solid rgba(129,140,248,0.5);
  background:rgba(129,140,248,0.18);
  color:#c7d2fe;
}

.suggestion-item{
  background:rgba(255,255,255,0.05);
  border-left:3px solid #818cf8;
  border-radius:10px;
  padding:10px 12px;
  margin:6px 0;
}

.small-muted{
  color:#a7b0c0;
  font-size:0.9rem;
}

.center-hero{
  min-height:58vh;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  text-align:center;
}
.center-hero h1{
  font-size:52px;
  margin-bottom:8px;
  color:#8B8BFF;
}
.center-hero p{
  font-size:18px;
  color:#9aa0a6;
  margin-bottom:18px;
}

.welcome-input-wrap{
  width:min(820px, 92%);
  margin:0 auto;
}

.stChatInput{
  position:fixed !important;
  bottom:18px;
  left:50%;
  transform:translateX(-50%);
  width:min(900px, calc(100% - 320px));
  z-index:9999;
  background:rgba(15,15,35,0.92);
  padding-top:10px;
}

.bottom-spacer{
  height:90px;
}

.mode-row{
  margin-bottom:12px;
}

.block-container{
  padding-bottom:110px !important;
}
</style>
""", unsafe_allow_html=True)


# ───────────────── MODEL LOADING ─────────────────
@st.cache_resource
def load_ser_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        return None
    scaler = TemperatureScaler()
    scaler.load(SCALER_PATH)
    return scaler

# ───────────────── SPEECH-TO-TEXT MODEL ─────────────────
@st.cache_resource
def load_whisper_model():
    """
    Loads faster-whisper speech recognition model.
    This converts user voice into text for chatbot understanding.
    """
    return WhisperModel("base")

def speech_to_text(audio_path):
    """
    Convert recorded voice to text.
    """
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path)
    text = " ".join([seg.text for seg in segments]).strip()
    return text


# ───────────────── AUDIO HELPERS ─────────────────
def record_audio(duration=5, sr=SR):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()

def load_audio_any(path_or_bytes, sr=SR):
    """
    Accepts a filesystem path OR bytes (wav/mp3/etc), returns (audio, sr)
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(path_or_bytes)
            tmp_path = tmp.name
        audio, _sr = librosa.load(tmp_path, sr=sr)
        return audio, _sr
    else:
        audio, _sr = librosa.load(path_or_bytes, sr=sr)
        return audio, _sr

def extract_audio_from_video(video_path):
    """
    Tries to extract audio from mp4/mkv/etc using moviepy if available.
    Returns a temp wav path.
    """
    try:
        import moviepy.editor as mp
    except Exception:
        return None, "moviepy not installed. Install: pip install moviepy"

    try:
        clip = mp.VideoFileClip(video_path)
        if clip.audio is None:
            return None, "No audio track found in the video."
        out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        clip.audio.write_audiofile(out_wav, fps=SR, verbose=False, logger=None)
        clip.close()
        return out_wav, None
    except Exception as e:
        return None, f"Video audio extraction failed: {e}"

def safe_plot_timeline(timeline_results, overall_emotion=None):
    """
    Safe wrapper to support different plot_timeline signatures.
    """
    try:
        return plot_timeline(timeline_results, overall_emotion=overall_emotion)
    except TypeError:
        try:
            return plot_timeline(timeline_results, overall_emotion)
        except TypeError:
            return plot_timeline(timeline_results)


def predict_emotion_from_audio(audio, sr, model, scaler):
    """
    Model expects (128, 40). We enforce MFCC only via FEATURE_TYPE_FIXED.
    """
    features = extract_features(audio, sr, feature_type=FEATURE_TYPE_FIXED)
    features = features[np.newaxis, ...]  # (1, 128, 40)

    probs = model.predict(features, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    emotion = LABELS[pred_class]

    if scaler:
        intensity, _ = scaler.get_intensity(probs, pred_class)
    else:
        intensity = int(float(probs[pred_class]) * 100)

    return emotion, int(intensity), probs, pred_class

def load_audio_from_url(url):
    """
    Downloads audio from URL using yt-dlp and returns waveform.
    Uses an absolute, verified path for FFmpeg.
    """
    import yt_dlp
    import os

    # 1. Get the absolute path to the 'app' folder
    # This ensures there is no confusion about relative directories
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Define the exact path to ffmpeg.exe to verify it exists
    ffmpeg_exe = os.path.join(app_dir, "ffmpeg.exe")
    
    if not os.path.exists(ffmpeg_exe):
        return None, None, f"ffmpeg.exe not found in {app_dir}. Please check file location."

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    ydl_opts = {
        'format': 'bestaudio/best',
        # Construct the options to be very specific
        'ffmpeg_location': ffmpeg_exe, 
        'outtmpl': temp_audio.replace(".wav", ".%(ext)s"),
        'quiet': True,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # yt-dlp converts the file and keeps the .wav extension
        wav_file = temp_audio.replace(".wav", ".wav")

        # Load the file into the SER model format
        # SR is your global sample rate (22050)
        audio, sr = librosa.load(wav_file, sr=SR)
        
        # Cleanup
        if os.path.exists(wav_file):
            os.remove(wav_file)
            
        return audio, sr, None

    except Exception as e:
        return None, None, str(e)

# ───────────────── SESSION STATE INIT ─────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if "songs_shown" not in st.session_state:
    st.session_state.songs_shown = False

if "songs" not in st.session_state:  # <─── FIXED: Added initialization
    st.session_state.songs = []

if "last_ser" not in st.session_state:
    # will store latest emotion/intensity/timeline/suggestions etc for sidebar
    st.session_state.last_ser = None

if "voice_reply" not in st.session_state:
    st.session_state.voice_reply = False

if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = "llama3.2"

if "chat_input_mode" not in st.session_state:
    st.session_state.chat_input_mode = "text"




# ───────────────── LOAD MODEL ─────────────────
model = load_ser_model()
scaler = load_scaler()

if model is None:
    st.error(f"Model not found. Make sure this exists: {MODEL_PATH}")
    st.stop()


# ───────────────── SIDEBAR (ANALYSIS PANEL) ─────────────────
with st.sidebar:
    st.title("EmotionGPT 🤖")
    st.markdown("---")

    st.markdown(f"**SER Model:** {'✅ Loaded' if os.path.exists(MODEL_PATH) else '❌ Missing'}")

    # AI model status
    if gemini_available():
        st.markdown("**AI Model:** 🤖 Gemini API Configured")
    elif check_ollama_available(st.session_state.ollama_model):
        st.markdown("**AI Model:** 🧠 Ollama Connected")
    else:
        st.markdown("**AI Model:** ⚠️ Using Rule-Based Chat")

    st.markdown("---")

    st.session_state.ollama_model = st.text_input(
        "Ollama Model",
        st.session_state.ollama_model
    )

    st.session_state.voice_reply = st.toggle(
        "🔊 Bot voice reply",
        value=st.session_state.voice_reply
    )

    st.caption("Browser voice uses gTTS. Offline local voice uses pyttsx3 if installed.")

    st.markdown("---")
    st.markdown("### 🎛 Analyzer Panel")

    if st.session_state.last_ser:
        ser = st.session_state.last_ser

        emo = ser.get("emotion", "Neutral")
        inten = ser.get("intensity", 0)
        summary = ser.get("summary", {})
        suggestions = ser.get("suggestions", [])
        fig = ser.get("timeline_fig", None)
        
        # Pull new songs into global session state if they exist in the analysis
        if ser.get("songs"):
            st.session_state.songs = ser.get("songs")

        # Emotion
        st.markdown(f"## {EMOTION_EMOJI.get(emo,'🎭')} {emo}")
        st.markdown(
            f"<span class='badge'>Estimated intensity: {inten}%</span>",
            unsafe_allow_html=True
        )

        # 🎵 SONGS
        st.markdown("---")
        st.markdown("### 🎵 Song Recommendations")

        if ser.get("songs"): st.session_state.songs = ser.get("songs")

        if st.session_state.songs:
            for title, link in st.session_state.songs:
                st.markdown(f"- [{title}]({link})")
        else:
            st.info("No songs yet — send voice/audio 🎧")

        # 📊 Timeline (only for audio)
        if fig:
            st.markdown("---")
            st.markdown("### 📊 Emotion Timeline")
            st.plotly_chart(fig, use_container_width=True)

        # 💡 Suggestions
        if suggestions:
            st.markdown("---")
            st.markdown("### 💡 Suggestions")
            for tip in suggestions[:6]:
                st.markdown(
                    f"<div class='suggestion-item'>{tip}</div>",
                    unsafe_allow_html=True
                )

        # 📈 Metrics
        if summary:
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Emotion", emo)
            c2.metric("Intensity", f"{summary.get('avg_intensity', inten):.0f}%")
            c3.metric("Trend", summary.get("intensity_trend", "stable").capitalize())

    else:
        st.info("No emotion detected yet. Start chatting to see analysis.")
# ───────────────── MAIN: TABS ─────────────────
tab_chat, tab_analyzer = st.tabs(["💬 Chat", "🎛 Analyzer (Manual)"])


# =================================================
# 1) CHAT TAB (DEFAULT EXPERIENCE)
# =================================================
with tab_chat:
    # Auto greeting (bot speaks first and must be visible first)
    if len(st.session_state.chat_history) == 0:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Hii Vaishu 💛 How are you yaar? Come on, tell me what’s going on with you today?"
        })

    # Create/update context based on latest SER if present
    if st.session_state.last_ser:
        ctx = {
            "emotion": st.session_state.last_ser["emotion"],
            "intensity": st.session_state.last_ser["intensity"],
            "duration": st.session_state.last_ser.get("summary", {}).get("duration", 0.0),
            "trend": st.session_state.last_ser.get("summary", {}).get("intensity_trend", "stable"),
            "top_emotions": ", ".join(
                list(st.session_state.last_ser.get("summary", {}).get("emotion_counts", {}).keys())[:3]
            ) or "Neutral",
            "language": "auto",
        }
    else:
        ctx = {
            "emotion": "Neutral",
            "intensity": 50,
            "duration": 0.0,
            "trend": "stable",
            "top_emotions": "Neutral",
            "language": "auto",
        }

    # Optional hero before first user message, but chat still visible
    only_greeting_present = (
        len(st.session_state.chat_history) == 1
        and st.session_state.chat_history[0]["role"] == "assistant"
    )

    if only_greeting_present:
        st.markdown("""
        <div class="center-hero">
          <h1>EmotionGPT</h1>
          <p>Ready for you all the time and I am listening.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Choose how you want to talk:")
    c1, c2, c3 = st.columns(3)

    chosen_input = None
    with c1:
        if st.button("⌨️ Text", use_container_width=True):
            chosen_input = "text"
    with c2:
        if st.button("🎤 Voice (record)", use_container_width=True):
            chosen_input = "mic"
    with c3:
        if st.button("📂 Upload Audio", use_container_width=True):
            chosen_input = "audio"
    

    if chosen_input:
        st.session_state.chat_input_mode = chosen_input
        st.rerun()

    st.markdown("---")

    # Always show chat history first
    for msg in st.session_state.chat_history:
        role_class = "chat-user" if msg["role"] == "user" else "chat-bot"
        prefix = "You" if msg["role"] == "user" else "Bot"
        st.markdown(
            f'<div class="{role_class}"><b>{prefix}:</b> {msg["content"]}</div>',
            unsafe_allow_html=True
        )

    mode = st.session_state.chat_input_mode
    

    # 1) TEXT
    if mode == "text":
        user_input = st.chat_input("Type here…")
    
        if user_input:
            st.session_state.chat_started = True

            # First append user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            history_for_llm = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ]

            response, _used_llm = chat(
                user_input,
                ctx,
                history_for_llm,
                st.session_state.ollama_model
            )

            # Get text emotion
            detected_emotion, detected_intensity = get_detected_text_emotion()
            ctx["emotion"] = detected_emotion
            ctx["intensity"] = detected_intensity

            # Update sidebar
            suggestions = get_suggestions(
                detected_emotion,
                detected_intensity,
                "stable"
            )

            songs = get_song_recos(detected_emotion) # Get songs for text mode too

            st.session_state.last_ser = {
                "emotion": detected_emotion,
                "intensity": detected_intensity,
                "songs": songs,
                "timeline_fig": None,
                "summary": {},
                "suggestions": suggestions,
            }

            # Songs (only once)
            if (
                not st.session_state.songs_shown
                and detected_emotion not in ["Neutral", "Calm"]
            ):
                song_text = "\n".join([f"🎵 [{t}]({l})" for t, l in songs])
                response += "\n\nOkay okay — mood playlist time 😌:\n" + song_text
                st.session_state.songs_shown = True

            # ✅ ALWAYS append bot reply
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

            if st.session_state.voice_reply:
                maybe_speak_text(response)
                play_bot_voice(response)

            st.rerun()

    # 2) MIC VOICE
    elif mode == "mic":
        st.markdown("Record a voice note, I’ll detect emotion from your audio 👇")
        dur = st.slider("Recording duration (seconds)", 2, 15, 5, key="chat_mic_dur")

        if st.button("🔴 Start recording", type="primary"):
            st.session_state.chat_started = True

            audio = record_audio(dur, SR)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio, SR)
                wav_path = tmp.name

            st.audio(wav_path)

            user_text = speech_to_text(wav_path)

            if user_text:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"🎤 {user_text}"
                })
            else:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"🎤 Voice message ({dur}s)"
                })

            emotion, intensity, probs, pred_class = predict_emotion_from_audio(audio, SR, model, scaler)

            timeline_results = analyze_timeline(
                audio, SR, model,
                lambda a, s: extract_features(a, s, feature_type=FEATURE_TYPE_FIXED),
                scaler,
                window_sec=1.0,
                hop_sec=0.5
            )
            summary = get_timeline_summary(timeline_results)
            suggestions = get_suggestions(emotion, intensity, summary.get("intensity_trend", "stable"))

            fig = None
            if timeline_results:
                fig = safe_plot_timeline(timeline_results, overall_emotion=emotion)

            songs = None
            if not st.session_state.songs_shown:
                songs = get_song_recos(emotion)
                st.session_state.songs_shown = True

            st.session_state.last_ser = {
                "emotion": emotion,
                "intensity": intensity,
                "songs": songs,
                "timeline_fig": fig,
                "summary": summary,
                "suggestions": suggestions
            }

            ctx = {
                "emotion": emotion,
                "intensity": intensity,
                "duration": summary.get("duration", 0.0),
                "trend": summary.get("intensity_trend", "stable"),
                "top_emotions": ", ".join(list(summary.get("emotion_counts", {}).keys())[:3]) or "Neutral",
                "language": "auto",
            }

            history_for_llm = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
            user_text_proxy = user_text if user_text else f"I sent a voice message. Detected emotion: {emotion} ({intensity}%)."
            response, _ = chat(user_text_proxy, ctx, history_for_llm, st.session_state.ollama_model)

            if songs:
                song_text = "\n".join([f"🎵 [{t}]({l})" for t, l in songs])
                response += "\n\nMood playlist time 😌:\n" + song_text

            st.session_state.chat_history.append({"role": "assistant", "content": response})

            if st.session_state.voice_reply:
                maybe_speak_text(response)
                play_bot_voice(response)

            st.rerun()

    # 3) UPLOAD AUDIO
    elif mode == "audio":
        up = st.file_uploader("Upload audio (.wav/.mp3/.ogg/.flac/.m4a)", type=["wav", "mp3", "ogg", "flac", "m4a"])
        if up:
            st.session_state.chat_started = True
            audio_bytes = up.read()
            audio, _sr = load_audio_any(audio_bytes, sr=SR)

            st.session_state.chat_history.append({"role": "user", "content": f"📎 Uploaded audio: {up.name}"})

            emotion, intensity, probs, pred_class = predict_emotion_from_audio(audio, SR, model, scaler)

            timeline_results = analyze_timeline(
                audio, SR, model,
                lambda a, s: extract_features(a, s, feature_type=FEATURE_TYPE_FIXED),
                scaler,
                window_sec=1.0,
                hop_sec=0.5
            )
            summary = get_timeline_summary(timeline_results)
            suggestions = get_suggestions(emotion, intensity, summary.get("intensity_trend", "stable"))

            fig = None
            if timeline_results:
                fig = safe_plot_timeline(timeline_results, overall_emotion=emotion)

            songs = None
            if not st.session_state.songs_shown:
                songs = get_song_recos(emotion)
                st.session_state.songs_shown = True

            st.session_state.last_ser = {
                "emotion": emotion,
                "intensity": intensity,
                "songs": songs,
                "timeline_fig": fig,
                "summary": summary,
                "suggestions": suggestions
            }

            ctx = {
                "emotion": emotion,
                "intensity": intensity,
                "duration": summary.get("duration", 0.0),
                "trend": summary.get("intensity_trend", "stable"),
                "top_emotions": ", ".join(list(summary.get("emotion_counts", {}).keys())[:3]) or "Neutral",
                "language": "auto",
            }

            history_for_llm = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
            user_text_proxy = f"I uploaded an audio file. Detected emotion: {emotion} ({intensity}%)."
            response, _ = chat(user_text_proxy, ctx, history_for_llm, st.session_state.ollama_model)

            if songs:
                song_text = "\n".join([f"🎵 [{t}]({l})" for t, l in songs])
                response += "\n\nMood playlist time 😌:\n" + song_text

            st.session_state.chat_history.append({"role": "assistant", "content": response})

            if st.session_state.voice_reply:
                maybe_speak_text(response)
                play_bot_voice(response)

            st.rerun()


    st.markdown('<div class="bottom-spacer"></div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🧹 Clear chat"):
        st.session_state.chat_history = []
        st.session_state.chat_started = False
        st.session_state.songs_shown = False
        st.session_state.songs = []
        st.session_state.last_ser = None
        st.session_state.chat_input_mode = "text"
        reset_session()
        st.rerun()


# =================================================
# 2) ANALYZER TAB (FINAL WORKING VERSION)
# =================================================
with tab_analyzer:
    st.header("🎛 Emotion Analyzer (Manual)")
    st.caption("Fast + Stable analyzer mode")

    st.markdown("**Feature Type:** mfcc (fixed for your trained model)")
    st.markdown("---")

    # ⚡ Persist audio across reruns
    if "analyzer_audio" not in st.session_state:
        st.session_state.analyzer_audio = None
        st.session_state.analyzer_sr = SR

    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False

    tab_up, tab_mic, tab_vid, tab_url = st.tabs([
        "📂 Upload Audio",
        "🎤 Record Mic",
        "🎬 Upload Video",
        "🌐 From URL"
    ])

    # ================= AUDIO UPLOAD =================
    with tab_up:
        uploaded = st.file_uploader(
            "Upload audio",
            type=["wav","mp3","ogg","flac","m4a"],
            key="an_up"
        )

        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded.read())
                path = tmp.name

            audio_data, audio_sr = librosa.load(path, sr=SR, duration=10.0)

            st.session_state.analyzer_audio = audio_data
            st.session_state.analyzer_sr = audio_sr

            st.audio(path)

    # ================= MIC =================
    with tab_mic:
        dur = st.slider("Recording duration", 2, 15, 5, key="an_mic_dur")

        if not st.session_state.is_recording:
            if st.button("🎤 Start Recording"):
                st.session_state.is_recording = True
                st.rerun()

        else:
            st.warning("🔴 Recording in progress...")

            audio_data = record_audio(dur, SR)

            st.session_state.analyzer_audio = audio_data
            st.session_state.analyzer_sr = SR

            st.session_state.is_recording = False

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio_data, SR)
                st.audio(tmp.name)

            st.success("Recording complete!")

    # ================= VIDEO =================
    with tab_vid:
        vid = st.file_uploader("Upload video", type=["mp4","mkv","mov"])

        if vid:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(vid.read())
                vid_path = tmp.name

            wav_path, err = extract_audio_from_video(vid_path)

            if err:
                st.error(err)
            else:
                audio_data, audio_sr = librosa.load(wav_path, sr=SR, duration=10.0)

                st.session_state.analyzer_audio = audio_data
                st.session_state.analyzer_sr = audio_sr

                st.success("Audio extracted from video!")

    # ================= URL =================
    with tab_url:
        url = st.text_input("🔗 Enter YouTube / audio URL")

        if st.button("Fetch from URL", key="fetch_url"):
            if url:
                with st.spinner("Downloading audio... 🎧"):
                    audio_data, audio_sr, err = load_audio_from_url(url)

                if err:
                    st.error(err)
                else:
                    st.session_state.analyzer_audio = audio_data
                    st.session_state.analyzer_sr = audio_sr
                    st.success("Audio loaded from URL!")

    # ================= ANALYZE =================
    if st.session_state.analyzer_audio is not None:
        if st.button("Analyze Emotion", type="primary"):

            with st.spinner("Analyzing emotion + building timeline... ⏳"):

                audio_data = st.session_state.analyzer_audio
                audio_sr = st.session_state.analyzer_sr

                # 🔥 LIMIT AUDIO (IMPORTANT)
                max_len = SR * 8
                if len(audio_data) > max_len:
                    audio_data = audio_data[:max_len]

                # 🔥 MAIN PREDICTION
                emotion, intensity, probs, pred_class = predict_emotion_from_audio(
                    audio_data, audio_sr, model, scaler
                )

                # 🔥 TIMELINE (FIXED)
                timeline_results = analyze_timeline(
                    audio_data,
                    audio_sr,
                    model,
                    lambda a, s: extract_features(a, s, feature_type=FEATURE_TYPE_FIXED),
                    scaler,
                    window_sec=1.0,
                    hop_sec=0.5
                )

                summary = get_timeline_summary(timeline_results)

                fig = None
                if timeline_results and len(timeline_results) > 0:
                    fig = safe_plot_timeline(
                        timeline_results,
                        overall_emotion=emotion
                    )

                suggestions = get_suggestions(
                    emotion,
                    intensity,
                    summary.get("intensity_trend","stable")
                )

                songs = get_song_recos(emotion)

                st.session_state.analyzer_result = {
                    "emotion": emotion,
                    "intensity": intensity,
                    "fig": fig,
                    "suggestions": suggestions,
                    "songs": songs
                }

            st.success("Analysis complete!")

    # ================= DISPLAY =================
    if "analyzer_result" in st.session_state:
        result = st.session_state.analyzer_result

        st.markdown("## 🎭 Detected Emotion")
        st.success(f"{result['emotion']} ({result['intensity']}%)")

        # ✅ GRAPH (THIS WAS MISSING)
        if result.get("fig") is not None:
            st.markdown("### 📊 Emotion Timeline")
            st.plotly_chart(result["fig"], use_container_width=True)

        # Suggestions
        if result["suggestions"]:
            st.markdown("### 💡 Suggestions")
            for tip in result["suggestions"]:
                st.write(f"- {tip}")

        # Songs
        if result["songs"]:
            st.markdown("### 🎵 Song Recommendations")
            for title, link in result["songs"]:
                st.markdown(f"- [{title}]({link})")