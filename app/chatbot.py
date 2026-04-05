"""
chatbot.py

Supports:
1) Gemini API (Google AI Studio) using the current Google GenAI SDK
2) Ollama local model (fallback if Gemini is not configured)
3) Rule-based fallback (if neither is available)

Features:
- Best-friend / companion style replies
- Emotion detection only from meaningful early prompts
- Session memory / personalized conversation context
- Cleaner fallback responses
- Greeting helper
- Song recommendations
- Optional offline TTS (pyttsx3)
- Interactive riddles (asks first, checks answer later)
"""

import os
import re
import random
import requests
from urllib.parse import quote_plus
from collections import Counter

from utils.suggestions import get_suggestions

# =========================================================
# CONFIG
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCC0ikn-4x2LlPbF12-qrded5Bb7p8PE30")
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "llama3.2"

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

SYSTEM_PROMPT_TEMPLATE = """You are Vaishu's close companion chatbot — warm, supportive, casual, emotionally aware, and natural.

Conversation context:
- Current emotion estimate: {emotion}
- Emotion intensity: {intensity}%
- Trend: {trend}
- Preferred language: {language}
- Known user details: {profile}
- Important memory notes: {memory_notes}
- Recent topics: {recent_topics}

How to respond:
- Sound warm, natural, human, and emotionally intelligent
- Keep replies short to medium
- Do NOT force emotion labels into every reply
- During the first few meaningful user prompts, be gently emotion-aware
- After that, continue like a normal close companion
- Use session memory naturally, without saying "I remember this is about..."
- Answer direct questions directly
- If user is playful, you can be playful
- If user is upset, be softer and comforting
- Match the user's language and tone when possible

Rules:
- DO NOT repeat phrases like "I'm listening", "tell me more", "go on"
- Avoid generic therapist-style replies
- Respond naturally like a friend
- You need to fine their emotin in two inputs from the user
- Be slightly casual and relatable
- If user mentions stress/work, acknowledge and respond practically
- Ask meaningful follow-up questions (not generic ones)

Emotion Awareness:
- If user is stressed: be supportive but not overly emotional
- If neutral: be engaging and curious
- If sad: be comforting
- If happy: be energetic

Safety:
- Do NOT give medical diagnoses
- Do NOT prescribe medication
- Do NOT claim the user has a disorder
- If user seems deeply distressed or unsafe, gently suggest reaching out to someone trusted or a professional

IMPORTANT:
Every reply must feel different and specific to the user's message.
"""

# =========================================================
# SONG RECOMMENDATIONS
# =========================================================

def _yt_search_link(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"

SONG_DB = {
    "Happy": [
        ("Butta Bomma (Telugu)", _yt_search_link("Butta Bomma song Telugu")),
        ("Samajavaragamana (Telugu)", _yt_search_link("Samajavaragamana song Telugu")),
        ("Arabic Kuthu (Tamil)", _yt_search_link("Arabic Kuthu song Tamil")),
    ],
    "Sad": [
        ("Inkem Inkem Inkem Kaavaale (Telugu)", _yt_search_link("Inkem Inkem Inkem Kaavaale song Telugu")),
        ("Adiga Adiga (Telugu)", _yt_search_link("Adiga Adiga song Telugu")),
        ("Why This Kolaveri Di (Tamil)", _yt_search_link("Why This Kolaveri Di song Tamil")),
    ],
    "Angry": [
        ("Saahore Baahubali (Telugu)", _yt_search_link("Saahore Baahubali Telugu song")),
        ("Jai Balayya (Telugu)", _yt_search_link("Jai Balayya song Telugu")),
        ("Surviva (Tamil)", _yt_search_link("Surviva song Tamil")),
    ],
    "Fearful": [
        ("Vellipomaakey (Telugu)", _yt_search_link("Vellipomaakey song Telugu")),
        ("Nee Kannu Neeli Samudram (Telugu)", _yt_search_link("Nee Kannu Neeli Samudram song Telugu")),
        ("Nenjame Nenjame (Tamil)", _yt_search_link("Nenjame Nenjame song Tamil")),
    ],
    "Calm": [
        ("Inthandham (Telugu)", _yt_search_link("Inthandham song Telugu")),
        ("Maate Vinadhuga (Telugu)", _yt_search_link("Maate Vinadhuga song Telugu")),
        ("Munbe Vaa (Tamil)", _yt_search_link("Munbe Vaa song Tamil")),
    ],
    "Neutral": [
        ("Oh Sita Hey Rama (Telugu)", _yt_search_link("Oh Sita Hey Rama Telugu")),
        ("The Life Of Ram (Telugu)", _yt_search_link("Life of Ram Telugu song")),
        ("Vaseegara (Tamil)", _yt_search_link("Vaseegara song Tamil")),
    ],
    "Disgust": [
        ("Naatu Naatu (Telugu)", _yt_search_link("Naatu Naatu Telugu song")),
        ("Pakka Local (Telugu)", _yt_search_link("Pakka Local Telugu song")),
        ("Appadi Podu (Tamil)", _yt_search_link("Appadi Podu Tamil song")),
    ],
    "Surprised": [
        ("Pataas Pilla (Telugu)", _yt_search_link("Pataas Pilla Telugu song")),
        ("Top Lesi Poddi (Telugu)", _yt_search_link("Top Lesi Poddi Telugu song")),
        ("Aaluma Doluma (Tamil)", _yt_search_link("Aaluma Doluma Tamil song")),
    ],
}

def get_song_recos(emotion: str):
    emotion = (emotion or "Neutral").strip().title()
    return SONG_DB.get(emotion, SONG_DB["Neutral"])

# =========================================================
# GREETING
# =========================================================

def get_initial_greeting(user_name="Vaishu"):
    greetings = [
        f"Hii {user_name} 💛 How are you yaar? Come on, tell me what’s going on with you today?",
        f"Hey {user_name} 😊 I’m here for you. How’s your day going?",
        f"Hii {user_name} ✨ Tell me, what’s on your mind today?",
    ]
    return random.choice(greetings)

# =========================================================
# TEXT EMOTION INFERENCE
# =========================================================

def is_meaningful_for_emotion(text: str) -> bool:
    if not text:
        return False

    low = text.lower().strip()
    tiny_msgs = {
        "hi", "hii", "hiii", "hello", "hey", "heyy",
        "ok", "okay", "hmm", "hmmm", "yo", "sup"
    }
    if low in tiny_msgs:
        return False

    if len(low.split()) < 4:
        return False

    return True

def infer_emotion_from_text(text: str):
    """
    Lightweight heuristic.
    Returns (emotion_label, intensity_0_to_100)
    """
    if not text:
        return "Neutral", 55

    t = text.lower().strip()
    intensity = 55

    if "!!!" in text or text.isupper():
        intensity = 80
    if len(text) > 160:
        intensity = min(90, intensity + 10)

    happy_k = ["happy", "excited", "awesome", "great", "good", "yay", "completed", "success", "won", "love it", "nice", "glad"]
    sad_k = ["sad", "cry", "upset", "hurt", "lonely", "tired", "break", "miss", "depressed", "bad", "low", "down"]
    angry_k = ["angry", "mad", "annoyed", "irritated", "hate", "frustrated", "furious"]
    fear_k = ["scared", "afraid", "worried", "anxious", "panic", "nervous", "fear", "stress", "stressed"]
    calm_k = ["calm", "relaxed", "peaceful", "fine", "chill"]
    disgust_k = ["disgust", "gross", "ew", "yuck", "nasty"]
    surprise_k = ["surprised", "shocked", "omg", "no way", "seriously", "unexpected"]

    def score(words):
        return sum(1 for w in words if w in t)

    scores = {
        "Happy": score(happy_k),
        "Sad": score(sad_k),
        "Angry": score(angry_k),
        "Fearful": score(fear_k),
        "Calm": score(calm_k),
        "Disgust": score(disgust_k),
        "Surprised": score(surprise_k),
    }

    best_emotion = max(scores, key=scores.get)
    best_score = scores[best_emotion]

    if best_score == 0:
        return "Neutral", intensity

    base_intensity = {
        "Happy": 70,
        "Sad": 68,
        "Angry": 74,
        "Fearful": 70,
        "Calm": 58,
        "Disgust": 66,
        "Surprised": 67,
    }

    return best_emotion, max(intensity, base_intensity.get(best_emotion, 60))

def aggregate_emotion_from_messages(messages):
    if not messages:
        return "Neutral", 55

    emotions = []
    intensities = []

    for msg in messages[:EMOTION_DETECTION_TURNS]:
        emo, inten = infer_emotion_from_text(msg)
        emotions.append(emo)
        intensities.append(inten)

    counter = Counter(emotions)
    final_emotion = counter.most_common(1)[0][0]
    matching = [intensities[i] for i, e in enumerate(emotions) if e == final_emotion]
    final_intensity = int(sum(matching) / len(matching))

    return final_emotion, final_intensity

# =========================================================
# MEMORY HELPERS
# =========================================================

def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def is_memory_worthy(msg: str) -> bool:
    low = msg.lower().strip()

    if len(low.split()) < 5:
        return False

    ignore_patterns = [
        "what can you do",
        "what could you do",
        "tell me what you can do",
        "who are you",
        "hi",
        "hello",
        "ok",
        "okay",
        "hmm",
    ]
    if any(p in low for p in ignore_patterns):
        return False

    memory_triggers = [
        "i feel", "i am", "i'm", "my", "today", "yesterday", "tomorrow",
        "because", "the problem is", "i want", "i need", "i like", "i love",
        "i enjoy", "i have", "i had"
    ]
    return any(trigger in low for trigger in memory_triggers)

def extract_user_profile(user_message: str):
    msg = _clean_text(user_message)
    low = msg.lower()

    name_match = re.search(r"\b(i am|i'm|my name is|call me)\s+([a-zA-Z][a-zA-Z ]{1,30})", msg, re.IGNORECASE)
    if name_match:
        possible_name = name_match.group(2).strip().split()[0]
        if possible_name:
            SESSION_STATE["user_profile"]["name"] = possible_name

    like_patterns = [
        r"\bi like ([^.!,\n]+)",
        r"\bi love ([^.!,\n]+)",
        r"\bi enjoy ([^.!,\n]+)",
    ]
    dislike_patterns = [
        r"\bi don't like ([^.!,\n]+)",
        r"\bi do not like ([^.!,\n]+)",
        r"\bi hate ([^.!,\n]+)",
    ]

    for pat in like_patterns:
        m = re.search(pat, low)
        if m:
            item = m.group(1).strip()
            if item and item not in SESSION_STATE["user_profile"]["likes"]:
                SESSION_STATE["user_profile"]["likes"].append(item)

    for pat in dislike_patterns:
        m = re.search(pat, low)
        if m:
            item = m.group(1).strip()
            if item and item not in SESSION_STATE["user_profile"]["dislikes"]:
                SESSION_STATE["user_profile"]["dislikes"].append(item)

    topic_keywords = [
        "exam", "college", "project", "family", "friend", "career", "stress",
        "health", "relationship", "interview", "job", "presentation", "emotion"
    ]
    for kw in topic_keywords:
        if kw in low and kw not in SESSION_STATE["user_profile"]["important_topics"]:
            SESSION_STATE["user_profile"]["important_topics"].append(kw)

def update_session_memory(user_message: str):
    msg = _clean_text(user_message)
    if not msg:
        return

    extract_user_profile(msg)

    if len(msg) > 20 and msg not in SESSION_STATE["last_topics"]:
        SESSION_STATE["last_topics"].append(msg[:120])

    SESSION_STATE["last_topics"] = SESSION_STATE["last_topics"][-6:]

    if is_memory_worthy(msg):
        if msg not in SESSION_STATE["memory_notes"]:
            SESSION_STATE["memory_notes"].append(msg)

    SESSION_STATE["memory_notes"] = SESSION_STATE["memory_notes"][-8:]

def build_profile_summary():
    profile = SESSION_STATE.get("user_profile", {})
    name = profile.get("name") or "Vaishu"
    likes = ", ".join(profile.get("likes", [])[:4]) if profile.get("likes") else "not specified"
    dislikes = ", ".join(profile.get("dislikes", [])[:4]) if profile.get("dislikes") else "not specified"
    topics = ", ".join(profile.get("important_topics", [])[:6]) if profile.get("important_topics") else "not specified"
    return f"name={name}; likes={likes}; dislikes={dislikes}; important_topics={topics}"

def build_memory_summary():
    notes = SESSION_STATE.get("memory_notes", [])[-5:]
    return " | ".join(notes) if notes else "none yet"

def build_recent_topics_summary():
    topics = SESSION_STATE.get("last_topics", [])[-4:]
    return " | ".join(topics) if topics else "none yet"

# =========================================================
# FUN EXTRAS
# =========================================================

JOKES = [
    "Okay listen 😭 If I had a rupee for every time I overthink… I’d be rich and still confused.",
    "Me: I’ll sleep early today 😌\nAlso me at 2:47 AM: Let’s replay every embarrassing moment of my life.",
    "I tried being productive today… but my bed said 'don’t leave me alone' 🥲",
    "Why do we open the fridge again and again like some new item will magically appear? 😭",
    "My brain during exam be like: loading… buffering… reconnecting… 🫠",
    "I don’t run away from problems… I just scroll past them 😌📱",
    "If laziness was a subject, I’d top the class without even studying 😎",
    "I opened my book to study and suddenly my brain remembered one awkward scene from 7 years ago 💀",
]

RIDDLES = [
    {
        "question": "Riddle time 😏 Oka illu undi... daniki talupu undadu, kitiki undadu. Lopala bangaram untundi. Adi enti?",
        "answers": ["egg", "guddu"]
    },
    {
        "question": "Cheppu chuddam 😌 Nannu entha teesthe antha pedda avthanu. Nenu enti?",
        "answers": ["hole", "gunta"]
    },
    {
        "question": "Idhi simple anukunta 👀 Naku kaallu levu kani nadustanu, naku noru ledu kani maatladuthanu. Adi enti?",
        "answers": ["clock", "gadiyaram"]
    },
    {
        "question": "Konchem alochinchi cheppu 😄 Nenu neellu lo puttanu, neellu lo periganu, kani neellu tagite chachipothanu. Adi enti?",
        "answers": ["fire", "agni"]
    },
    {
        "question": "Try cheyyi 😏 Oka chettu meeda pannendu kommulu, prati kommaki muppai akulu. Adi enti?",
        "answers": ["calendar", "samvatsaram calendar"]
    },
    {
        "question": "Idhi konchem fun 😌 Nenu padukunte nilchunta, nilchunte padukunta. Nenu enti?",
        "answers": ["footwear", "slippers", "shoe", "shoes", "cheppulu"]
    },
    {
        "question": "Sare idi cheppu 👀 Nannu kotte varaku nenu maatladanu. Kottagane sound chestanu. Adi enti?",
        "answers": ["bell", "ganta"]
    },
    {
        "question": "Final ga idi 😄 Ratri vastundi, pagalu pothundi. Pattukolevu, kani chustaru. Adi enti?",
        "answers": ["darkness", "cheekati"]
    },
]

def check_riddle_answer(user_message: str):
    pending = SESSION_STATE.get("pending_riddle")
    if not pending:
        return None

    user_ans = user_message.lower().strip()
    correct_answers = pending.get("answers", [])

    if isinstance(correct_answers, str):
        correct_answers = [correct_answers]

    SESSION_STATE["pending_riddle"] = None

    if any(ans.lower() in user_ans for ans in correct_answers):
        return random.choice([
            "Yesss correct 😄🔥 Super cheppav!",
            "Ayyoo nicee 😍 Correct answer cheppesaav!",
            "Perfecttt 😌 Nuvvu riddle champ anipisthundi!",
        ])
    else:
        reveal = correct_answers[0].capitalize() if correct_answers else "I couldn’t find the answer"
        return random.choice([
            f"Ahh close kaani kaadhu 😅 Correct answer: **{reveal}**",
            f"Hehe almost 😄 Correct answer enti ante: **{reveal}**",
            f"Not bad ra 😌 But correct answer is: **{reveal}**",
        ])

def should_add_fun(user_message: str) -> bool:
    low = user_message.lower()
    serious_markers = [
        "sad", "stress", "stressed", "worried", "upset", "problem",
        "help", "can you", "what can you do", "what could you do",
        "explain", "how", "why"
    ]
    if any(m in low for m in serious_markers):
        return False
    return True

def maybe_add_fun(text: str, user_message: str) -> str:
    if not should_add_fun(user_message):
        return text

    roll = random.random()
    if roll < 0.08:
        return text + "\n\n" + random.choice(JOKES)
    return text

# =========================================================
# OLLAMA
# =========================================================

def check_ollama_available(model=DEFAULT_OLLAMA_MODEL):
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any(model.split(":")[0] in m for m in models)
    except Exception:
        pass
    return False

def chat_with_ollama(user_message, context, history=None, model=DEFAULT_OLLAMA_MODEL):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(**context)

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.82, "top_p": 0.9},
    }

    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()["message"]["content"], True
    except Exception:
        pass

    return None, False

# =========================================================
# GEMINI
# =========================================================

def gemini_available() -> bool:
    bad_values = {"", "PASTE_YOUR_KEY_HERE", "YOUR_GEMINI_API_KEY", "YOUR_API_KEY", None}
    return GEMINI_API_KEY not in bad_values

def chat_with_gemini(user_message, context, history=None, model=DEFAULT_GEMINI_MODEL):
    if not gemini_available():
        return None, False

    try:
        from google import genai
    except Exception:
        return None, False

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(**context)

        history_text = ""
        if history:
            for m in history[-10:]:
                role = m.get("role", "user").upper()
                content = m.get("content", "")
                history_text += f"{role}: {content}\n"

        prompt = (
            f"{system_prompt}\n\n"
            f"Conversation so far:\n{history_text}\n"
            f"USER: {user_message}\n"
            f"ASSISTANT:"
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if text:
            return text.strip(), True
        return None, False

    except Exception as e:
        print("Gemini error:", e)
        return None, False

# =========================================================
# RULE-BASED FALLBACK
# =========================================================

EARLY_EMOTION_RESPONSES = [
    "I’m here with you 💛 Tell me a little more — I want to understand properly.",
    "Hmm okay, I’m listening carefully. Tell me more?",
    "Got you 🫶 Go on, I’m here for you.",
]

GENERAL_RESPONSES = {
    "what_can_you_do": "Of course 💛 I can chat with you like a companion, respond in a warm personal way, understand your mood from meaningful early messages, remember things during this session, suggest songs, and also talk normally about anything you want.",
    "who_are_you": "I’m your companion-style chatbot 💛 You can talk to me casually, vent, ask questions, or just chat randomly. I’m here for all of it.",
}

NORMAL_COMPANION_RESPONSES = [
    "I’m here, tell me everything 💛",
    "Hmm okay, I’m listening properly 👀",
    "Got you. Go on, I’m with you.",
    "Okayy, tell me more 😌",
]

def fallback_chat(user_message, context):
    low = user_message.lower().strip()

    # If user is answering a pending riddle
    riddle_reply = check_riddle_answer(user_message)
    if riddle_reply:
        return riddle_reply, False

    if "what can you do" in low or "what could you do" in low:
        return GENERAL_RESPONSES["what_can_you_do"], False

    if "who are you" in low:
        return GENERAL_RESPONSES["who_are_you"], False

    if "joke" in low:
        return random.choice(JOKES), False

    if "riddle" in low or "podupu katha" in low:
        r = random.choice(RIDDLES)
        SESSION_STATE["pending_riddle"] = r
        return r["question"], False

    emotion = context.get("emotion", "Neutral")
    intensity = int(context.get("intensity", 50))
    tips = get_suggestions(emotion, intensity, context.get("trend", "stable"))
    tip = tips[0] if tips else "Take one slow breath first… I’m here."

    meaningful_count = len(SESSION_STATE["meaningful_user_messages"])

    if meaningful_count < EMOTION_DETECTION_TURNS:
        txt = random.choice(EARLY_EMOTION_RESPONSES)
    else:
        base = random.choice(NORMAL_COMPANION_RESPONSES)
        txt = base
        if SESSION_STATE["memory_notes"]:
            txt += " You can continue from where you left off — I’m following."

    if emotion not in ["Neutral", "Calm"] and meaningful_count <= EMOTION_DETECTION_TURNS:
        txt += f" {tip}"

    return maybe_add_fun(txt, user_message), False

# =========================================================
# CONTEXT BUILDERS
# =========================================================

def merge_external_context(context):
    context = context or {}

    detected_emotion = SESSION_STATE.get("detected_emotion", "Neutral")
    detected_intensity = SESSION_STATE.get("detected_intensity", 55)

    merged = {
        "emotion": context.get("emotion", detected_emotion) or detected_emotion,
        "intensity": context.get("intensity", detected_intensity) or detected_intensity,
        "duration": context.get("duration", 0.0),
        "trend": context.get("trend", "stable"),
        "top_emotions": context.get("top_emotions", "Not enough data"),
        "language": context.get("language", "English"),
        "profile": build_profile_summary(),
        "memory_notes": build_memory_summary(),
        "recent_topics": build_recent_topics_summary(),
    }
    return merged

def maybe_update_emotion_state(user_message, history=None):
    if SESSION_STATE["emotion_locked"]:
        return

    if not is_meaningful_for_emotion(user_message):
        return

    if user_message not in SESSION_STATE["meaningful_user_messages"]:
        SESSION_STATE["meaningful_user_messages"].append(user_message)

    limited_msgs = SESSION_STATE["meaningful_user_messages"][:EMOTION_DETECTION_TURNS]
    emo, inten = aggregate_emotion_from_messages(limited_msgs)

    SESSION_STATE["detected_emotion"] = emo
    SESSION_STATE["detected_intensity"] = inten
    SESSION_STATE["emotion_checks_done"] = len(limited_msgs)

    if len(limited_msgs) >= EMOTION_DETECTION_TURNS:
        SESSION_STATE["emotion_locked"] = True

# =========================================================
# MAIN CHAT API
# =========================================================

def chat(user_message, context, history=None, model=DEFAULT_OLLAMA_MODEL):
    if not SESSION_STATE["greeted"]:
        SESSION_STATE["greeted"] = True

    update_session_memory(user_message)
    maybe_update_emotion_state(user_message, history)

    merged_context = merge_external_context(context)

    if gemini_available():
        response, ok = chat_with_gemini(user_message, merged_context, history)
        if ok and response:
            return maybe_add_fun(response, user_message), True

    if check_ollama_available(model):
        response, ok = chat_with_ollama(user_message, merged_context, history, model)
        if ok and response:
            return maybe_add_fun(response, user_message), True

    return fallback_chat(user_message, merged_context)

# =========================================================
# OPTIONAL HELPERS FOR APP.PY
# =========================================================

def get_or_create_greeting(user_name="Vaishu"):
    if not SESSION_STATE["greeted"]:
        SESSION_STATE["greeted"] = True
        stored_name = SESSION_STATE["user_profile"].get("name") or user_name
        return get_initial_greeting(stored_name)
    return None

def get_detected_text_emotion():
    return SESSION_STATE.get("detected_emotion", "Neutral"), SESSION_STATE.get("detected_intensity", 55)

def reset_session():
    SESSION_STATE["greeted"] = False
    SESSION_STATE["emotion_locked"] = False
    SESSION_STATE["emotion_checks_done"] = 0
    SESSION_STATE["detected_emotion"] = "Neutral"
    SESSION_STATE["detected_intensity"] = 55
    SESSION_STATE["meaningful_user_messages"] = []
    SESSION_STATE["pending_riddle"] = None
    SESSION_STATE["user_profile"] = {
        "name": None,
        "likes": [],
        "dislikes": [],
        "important_topics": [],
    }
    SESSION_STATE["memory_notes"] = []
    SESSION_STATE["last_topics"] = []

# =========================================================
# VOICE OUTPUT HELPERS
# =========================================================

def speak_out_loud(text: str):
    try:
        import pyttsx3
    except Exception:
        return False, "pyttsx3 not installed"

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(text)
        engine.runAndWait()
        return True, None
    except Exception as e:
        return False, str(e)

def maybe_speak_text(text: str):
    ok, _err = speak_out_loud(text)
    return ok

# =========================================================
# TEXT TO SPEECH (gTTS)
# =========================================================

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