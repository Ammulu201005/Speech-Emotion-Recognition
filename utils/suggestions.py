"""
Rule-based suggestion engine.
Based on dominant emotion + intensity trend → 3-5 suggestions.
Non-medical. Includes safe coping tips, quotes, music genre suggestions.
"""

SUGGESTIONS_DB = {
    'Happy': {
        'low':  ["Keep that positive energy — share it with someone today.",
                 "Channel this good mood into a creative project.",
                 "Music suggestion: Upbeat pop or jazz to match your vibe.",
                 "Quote: \"Happiness is not something ready-made. It comes from your own actions.\" – Dalai Lama"],
        'high': ["You're radiating great energy! Consider expressing gratitude.",
                 "Great time to tackle a challenge or start something new.",
                 "Music suggestion: Feel-good classics or dance music.",
                 "Share your positivity — call a friend!",
                 "Quote: \"The most wasted day is one without laughter.\" – Nicolas Chamfort"],
    },
    'Sad': {
        'low':  ["It's okay to feel sad — allow yourself to process it.",
                 "Try a short walk outside to gently shift your mood.",
                 "Music suggestion: Acoustic/indie folk for a gentle lift.",
                 "Quote: \"Even the darkest night will end and the sun will rise.\" – Victor Hugo"],
        'high': ["Reach out to someone you trust — talking helps.",
                 "Try journaling your thoughts for 5 minutes.",
                 "Music suggestion: Calming classical or ambient music.",
                 "A warm drink and a comfortable space can help.",
                 "If you feel persistently low, consider talking to a counselor. You're not alone."],
    },
    'Angry': {
        'low':  ["Take 3 slow deep breaths — it genuinely helps.",
                 "Physical movement (a brisk walk) can release tension.",
                 "Music suggestion: Rock or metal can be a healthy emotional outlet.",
                 "Quote: \"For every minute you are angry you lose sixty seconds of peace.\" – Ralph Waldo Emerson"],
        'high': ["Step away from the situation for a few minutes before responding.",
                 "Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s.",
                 "Music suggestion: Drumming or high-energy workout music for release.",
                 "Write down what triggered you — it helps identify patterns.",
                 "If anger feels overwhelming frequently, speaking to a professional can help."],
    },
    'Fearful': {
        'low':  ["Name what you're anxious about — awareness reduces fear.",
                 "Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you touch...",
                 "Music suggestion: Ambient or lo-fi music to calm your nervous system.",
                 "Quote: \"You gain strength, courage, and confidence by every experience in which you really stop to look fear in the face.\" – Eleanor Roosevelt"],
        'high': ["Focus on slow, controlled breathing right now.",
                 "Remind yourself: this moment will pass.",
                 "Music suggestion: Nature sounds or binaural beats for deep calm.",
                 "Ground yourself in the present — feel your feet on the floor.",
                 "If anxiety is intense or frequent, please reach out to a mental health professional."],
    },
    'Disgust': {
        'low':  ["Take a moment to step away from what's bothering you.",
                 "Identify exactly what triggered the feeling — it helps process it.",
                 "Music suggestion: Upbeat pop to shift your focus.",
                 "Quote: \"In the middle of every difficulty lies opportunity.\" – Albert Einstein"],
        'high': ["Setting boundaries is healthy — it's okay to say no to things that disturb you.",
                 "Try refocusing on something you genuinely enjoy.",
                 "Music suggestion: Uplifting or humorous podcasts/music for a mood shift.",
                 "Talk to a friend about what's bothering you.",
                 "If something deeply uncomfortable is recurrent, journaling may help clarify your feelings."],
    },
    'Neutral': {
        'low':  ["A neutral state is a great base — use it to plan or reflect.",
                 "This is a good time for focused work or learning.",
                 "Music suggestion: Instrumental or lo-fi for focused productivity.",
                 "Quote: \"Peace is not the absence of conflict, but the presence of creative alternatives.\" – Dorothy Thompson"],
        'high': ["You're in balance — make the most of this calm state.",
                 "Try meditation or mindful breathing to deepen the stillness.",
                 "Music suggestion: Classical or jazz for a calm, alert mind.",
                 "Plan something you've been putting off."],
    },
    'Calm': {
        'low':  ["Maintain this peaceful state with a gentle walk.",
                 "Great time for reflection or journaling.",
                 "Music suggestion: Acoustic or ambient sounds.",
                 "Quote: \"Calmness is the cradle of power.\" – Josiah Gilbert Holland"],
        'high': ["Deep calm is a gift — consider meditating to extend it.",
                 "Share this energy by helping someone around you.",
                 "Music suggestion: Spa or nature soundscapes.",
                 "Use this time for creative or thoughtful activities."],
    },
    'Surprised': {
        'low':  ["Embrace the unexpected — curiosity is a superpower.",
                 "Reflect on what surprised you and what you can learn from it.",
                 "Music suggestion: Jazz or experimental music for open-minded vibes.",
                 "Quote: \"The moment of surprise is a moment of learning.\" – Unknown"],
        'high': ["Take a breath and let the surprise settle before reacting.",
                 "Channel that energy — it can spark creativity!",
                 "Music suggestion: Upbeat, energetic music to match the energy.",
                 "Surprises often lead to growth — stay open.",
                 "Share what surprised you with a friend — storytelling helps process emotion."],
    },
}

CRISIS_EMOTIONS = {'Fearful', 'Sad', 'Angry'}
CRISIS_INTENSITY_THRESHOLD = 75


def get_suggestions(emotion, intensity_pct, trend='neutral'):
    """
    Generate 3-5 suggestions based on emotion + intensity level.
    """
    db = SUGGESTIONS_DB.get(emotion, SUGGESTIONS_DB['Neutral'])
    level = 'high' if intensity_pct >= 50 else 'low'
    tips = db.get(level, db.get('low', []))[:4]

    # Add trend-based tip
    if trend == 'rising' and intensity_pct > 60:
        tips.append("Your intensity is increasing — this might be a good time for a short break.")
    elif trend == 'falling':
        tips.append("Your emotional intensity is easing — keep going, you're doing well.")

    # Crisis note
    if emotion in CRISIS_EMOTIONS and intensity_pct >= CRISIS_INTENSITY_THRESHOLD:
        tips.append(
            "⚠️ Note: If you're feeling overwhelmed, please reach out to a trusted person "
            "or a mental health helpline. You don't have to go through this alone."
        )

    return tips[:5]
