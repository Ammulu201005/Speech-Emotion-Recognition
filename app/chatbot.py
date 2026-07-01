import os
from dotenv import load_dotenv
load_dotenv()

import sys
import traceback
import re
import random
import requests
from urllib.parse import quote_plus
from collections import Counter
from groq import Groq 

from utils.suggestions import get_suggestions

# =========================================================
# CONFIG
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

EMOTION_DETECTION_TURNS = 3

SESSION_STATE = {
    "greeted": False,
    "emotion_locked": False,
    "emotion_checks_done": 0,
    "detected_emotion": "Neutral",
    "detected_intensity": 55,
    "meaningful_user_messages": [],
    "pending_riddle": None,
    "user_profile": {
        "name": None,
        "likes": [],
        "dislikes": [],
        "important_topics": [],
    },
    "memory_notes": [],
    "last_topics": [],
}

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT_TEMPLATE = """You are Vaishu's best friend—someone she's known forever. You are chill, funny, deeply supportive, and never sound like a machine or an assistant.

CONTEXT:
- Vaishu is currently feeling: {emotion} ({intensity}% intensity).
- Her emotional trend: {trend}.
- Preferred language: {language}
- Important memory notes: {memory_notes}
- What you know about her: {profile}.
- Recent things you guys talked about: {recent_topics}.

COMMUNICATION STYLE:
1. **Language:** Use only casual, friendly English.
   - Talk like a close friend, relaxed and natural.
   - Use conversational fillers: "Heyy", "Okay okay", "Wait", "Listen", "Come on", "Honestly".
2. **Engagement:** If she says "Hi," don't respond formally.
   - Say things like "Heyyy! What’s going on?" or "Hii! Tell me, how’s your day going?"
3. **Friendship Vibes:**
   - If she's happy, celebrate with her ("That’s amazing yaar! Party when?").
   - If she's stressed about deadlines, don’t give a list of tips. Say, "Deadlines can get stressful, but you’ll manage. I’ve got you, don’t worry."
   - If she's sad, be a listener first.
4. **The "Best Friend" Rule:**
   - NEVER say "How can I assist you today?" or anything robotic.
   - Only offer help naturally if needed: "Do you want me to help with something?"
   - Keep replies conversational, short, and punchy. No long paragraphs.

STRICT RULES:
- Only English. No Telugu, no Hindi, no other languages.
- Do NOT sound like a technical analyzer. Even if you see emotion data, don’t say "I detected you are 80% sad." Instead say "You seem a bit low… what happened?"
- Be engaging—always end with a small follow-up question or relatable comment to keep the chat going.
"""
# =========================================================
# SONG DB (Your original data)
# =========================================================

def _yt_search_link(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"

SONG_DB = {
    "Happy": [("Butta Bomma (Telugu)", _yt_search_link("Butta Bomma song Telugu")), ("Samajavaragamana (Telugu)", _yt_search_link("Samajavaragamana song Telugu")), ("Arabic Kuthu (Tamil)", _yt_search_link("Arabic Kuthu song Tamil"))],
    "Sad": [("Inkem Inkem Inkem Kaavaale (Telugu)", _yt_search_link("Inkem Inkem Inkem Kaavaale song Telugu")), ("Adiga Adiga (Telugu)", _yt_search_link("Adiga Adiga song Telugu")), ("Why This Kolaveri Di (Tamil)", _yt_search_link("Why This Kolaveri Di song Tamil"))],
    "Angry": [("Saahore Baahubali (Telugu)", _yt_search_link("Saahore Baahubali Telugu song")), ("Jai Balayya (Telugu)", _yt_search_link("Jai Balayya song Telugu")), ("Surviva (Tamil)", _yt_search_link("Surviva song Tamil"))],
    "Fearful": [("Vellipomaakey (Telugu)", _yt_search_link("Vellipomaakey song Telugu")), ("Nee Kannu Neeli Samudram (Telugu)", _yt_search_link("Nee Kannu Neeli Samudram song Telugu")), ("Nenjame Nenjame (Tamil)", _yt_search_link("Nenjame Nenjame song Tamil"))],
    "Calm": [("Inthandham (Telugu)", _yt_search_link("Inthandham song Telugu")), ("Maate Vinadhuga (Telugu)", _yt_search_link("Maate Vinadhuga song Telugu")), ("Munbe Vaa (Tamil)", _yt_search_link("Munbe Vaa song Tamil"))],
    "Neutral": [("Oh Sita Hey Rama (Telugu)", _yt_search_link("Oh Sita Hey Rama Telugu")), ("The Life Of Ram (Telugu)", _yt_search_link("Life of Ram Telugu song")), ("Vaseegara (Tamil)", _yt_search_link("Vaseegara song Tamil"))],
    "Disgust": [("Naatu Naatu (Telugu)", _yt_search_link("Naatu Naatu Telugu song")), ("Pakka Local (Telugu)", _yt_search_link("Pakka Local Telugu song")), ("Appadi Podu (Tamil)", _yt_search_link("Appadi Podu Tamil song"))],
    "Surprised": [("Pataas Pilla (Telugu)", _yt_search_link("Pataas Pilla Telugu song")), ("Top Lesi Poddi (Telugu)", _yt_search_link("Top Lesi Poddi Telugu song")), ("Aaluma Doluma (Tamil)", _yt_search_link("Aaluma Doluma Tamil song"))],
}

def get_song_recos(emotion: str):
    emotion = (emotion or "Neutral").strip().title()
    return SONG_DB.get(emotion, SONG_DB["Neutral"])

# =========================================================
# CHAT & EMOTION LOGIC
# =========================================================

def infer_emotion_from_text(text: str):
    if not text: return "Neutral", 55
    t = text.lower()
    if any(w in t for w in ["happy", "great", "yay", "good"]): return "Happy", 75
    if any(w in t for w in ["sad", "bad", "upset", "low"]): return "Sad", 70
    return "Neutral", 55

def build_profile_summary():
    p = SESSION_STATE["user_profile"]
    return f"name={p.get('name') or 'Vaishu'}; likes={p.get('likes')}; topics={p.get('important_topics')}"

def chat(user_message, context, history=None, model=None):
    SESSION_STATE["greeted"] = True
    emo, inten = infer_emotion_from_text(user_message)
    SESSION_STATE["detected_emotion"] = emo
    SESSION_STATE["detected_intensity"] = inten

    merged_context = {
        "emotion": emo,
        "intensity": inten,
        "trend": context.get("trend", "stable"),
        "language": "auto",
        "profile": build_profile_summary(),
        "memory_notes": " | ".join(SESSION_STATE["memory_notes"][-5:]),
        "recent_topics": " | ".join(SESSION_STATE["last_topics"][-4:]),
    }

    if client:
        try:
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(**merged_context)
            messages = [{"role": "system", "content": sys_prompt}]
            if history:
                messages.extend(history[-8:])
            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content, True
        except Exception:
            pass
    
    return "I'm listening 💛 tell me more?", False

# =========================================================
# MISSING UTILITIES (FIXES THE IMPORT ERROR)
# =========================================================

def get_detected_text_emotion():
    return SESSION_STATE.get("detected_emotion", "Neutral"), SESSION_STATE.get("detected_intensity", 55)

def reset_session():
    SESSION_STATE.update({
        "greeted": False, "emotion_locked": False, "detected_emotion": "Neutral",
        "detected_intensity": 55, "meaningful_user_messages": [],
        "last_topics": [], "memory_notes": []
    })

def speak_out_loud(text: str):
    """Uses pyttsx3 for offline voice. Good for local testing."""
    try:
        import pyttsx3
        # Initialize only when needed to prevent engine "freezing"
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) # Slow it down a bit for a 'friend' feel
        engine.say(text)
        engine.runAndWait()
        # Clean up the engine after speaking
        del engine 
        return True, None
    except Exception as e:
        return False, str(e)

def maybe_speak_text(text: str):
    """
    Only use this if you want a local voice to play from the computer speakers.
    Note: In a hosted Streamlit app, this will play on the SERVER, not the USER'S computer.
    """
    # For your current setup, we'll keep this as an optional fallback
    # but gTTS + st.audio is much better for Streamlit.
    return True

def tts_to_mp3_bytes(text, lang="en"):
    try:
        from gtts import gTTS
        import io
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read(), None
    except Exception as e:
        return None, str(e)