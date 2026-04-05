"""
Feature extraction: MFCC, ZCR, RMS, Chroma, Mel
Best per paper: MFCC alone for most datasets.
Improved: MFCC + Delta + Delta-Delta (temporal context).
"""
import numpy as np
import librosa

N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048


def extract_mfcc(audio, sr, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)  # (120, T)


def extract_zcr(audio, sr):
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    return zcr  # (1, T)


def extract_rms(audio, sr):
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    return rms  # (1, T)


def extract_chroma(audio, sr):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return chroma  # (12, T)


def extract_mel(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # (128, T)


def pad_or_truncate(feature, max_len=128):
    """Pad or truncate time dimension to fixed length."""
    if feature.shape[1] > max_len:
        return feature[:, :max_len]
    pad_width = max_len - feature.shape[1]
    return np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')


def extract_features(audio, sr, feature_type='mfcc', max_len=128):
    """
    feature_type: 'mfcc' | 'mfcc_delta' | 'all' | 'zcr' | 'rms' | 'chroma' | 'mel'
    Returns flat 1D vector for CNN input.
    """
    if feature_type == 'mfcc':
        feat = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    elif feature_type == 'mfcc_delta':
        feat = extract_mfcc(audio, sr)
    elif feature_type == 'all':
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        zcr = extract_zcr(audio, sr)
        rms = extract_rms(audio, sr)
        chroma = extract_chroma(audio, sr)
        feat = np.vstack([mfcc, zcr, rms, chroma])
    elif feature_type == 'mel':
        feat = extract_mel(audio, sr)
    elif feature_type == 'zcr':
        feat = extract_zcr(audio, sr)
    elif feature_type == 'rms':
        feat = extract_rms(audio, sr)
    elif feature_type == 'chroma':
        feat = extract_chroma(audio, sr)
    else:
        feat = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    feat = pad_or_truncate(feat, max_len=max_len)
    return feat.T  # Return (max_len, n_features) for LSTM compatibility
