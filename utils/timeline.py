"""
Timeline analysis: split audio into windows, predict per window.
Window: 1s, Hop: 0.5s (configurable)
Returns emotion labels + intensity per window for plotting.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LABELS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

EMOTION_COLORS = {
    'Neutral': '#94a3b8', 'Calm': '#60a5fa', 'Happy': '#fbbf24',
    'Sad': '#818cf8', 'Angry': '#f87171', 'Fearful': '#a78bfa',
    'Disgust': '#34d399', 'Surprised': '#fb923c'
}


def analyze_timeline(audio, sr, model, feature_fn, temperature_scaler,
                     window_sec=1.0, hop_sec=0.5):
    """
    Slide a window over audio and predict emotion+intensity per window.
    Returns: list of dicts with keys: time_start, time_end, emotion, intensity, probs
    """
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    results = []

    if window_samples <= 0 or hop_samples <= 0:
        return results
    if len(audio) < window_samples:
        return results

    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        segment = audio[start:start + window_samples]
        features = feature_fn(segment, sr)          # (T, F)
        features = features[np.newaxis, ...]        # (1, T, F)

        probs = model.predict(features, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        emotion = LABELS[pred_class]

        if temperature_scaler is not None:
            intensity, _conf = temperature_scaler.get_intensity(probs, pred_class)
        else:
            intensity = int(float(probs[pred_class]) * 100)

        results.append({
            'time_start': start / sr,
            'time_end': (start + window_samples) / sr,
            'time_mid': (start + window_samples / 2) / sr,
            'emotion': emotion,
            'intensity': int(intensity),
            'probs': probs.tolist()
        })

    return results


def plot_timeline(results, overall_emotion=None, **kwargs):
    """Create interactive Plotly timeline figure."""
    if not results:
        return None

    times = [r['time_mid'] for r in results]
    emotions = [r['emotion'] for r in results]
    intensities = [r['intensity'] for r in results]
    colors = [EMOTION_COLORS.get(e, '#888') for e in emotions]

    title_overall = overall_emotion if overall_emotion else "—"

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Emotion Timeline (Overall: {title_overall})',
                        'Intensity Over Time (Estimated)'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )

    # Emotion scatter
    fig.add_trace(go.Scatter(
        x=times, y=emotions,
        mode='markers+lines',
        marker=dict(color=colors, size=12, line=dict(width=1, color='white')),
        line=dict(color='rgba(255,255,255,0.2)', width=1),
        text=[f"t={t:.1f}s<br>{e}<br>Intensity: {i}%" for t, e, i in zip(times, emotions, intensities)],
        hoverinfo='text',
        name='Emotion'
    ), row=1, col=1)

    # Intensity line
    fig.add_trace(go.Scatter(
        x=times, y=intensities,
        mode='lines+markers',
        line=dict(color='#818cf8', width=2),
        marker=dict(size=6, color=colors),
        fill='tozeroy',
        fillcolor='rgba(129,140,248,0.15)',
        name='Intensity %'
    ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(15,15,35,0.95)',
        plot_bgcolor='rgba(15,15,35,0.7)',
        font=dict(color='white'),
        showlegend=False,
        height=480,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    fig.update_xaxes(title_text='Time (s)', row=2, col=1)
    fig.update_yaxes(title_text='Emotion', row=1, col=1)
    fig.update_yaxes(title_text='Intensity (%)', range=[0, 105], row=2, col=1)

    return fig


def get_timeline_summary(results):
    """Summarize timeline results: dominant emotion, trend, etc."""
    if not results:
        return {}

    emotions = [r['emotion'] for r in results]
    from collections import Counter
    counts = Counter(emotions)

    dominant = counts.most_common(1)[0][0]
    intensities = [r['intensity'] for r in results]
    avg_intensity = float(np.mean(intensities))

    first_half = intensities[:len(intensities)//2] or intensities
    second_half = intensities[len(intensities)//2:] or intensities

    trend = 'rising' if float(np.mean(second_half)) > float(np.mean(first_half)) else 'falling'

    return {
        'dominant_emotion': dominant,
        'emotion_counts': dict(counts),
        'avg_intensity': avg_intensity,
        'intensity_trend': trend,
        'duration': results[-1]['time_end']
    }