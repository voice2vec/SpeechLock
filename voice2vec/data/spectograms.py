import numpy as np
from librosa import load, logamplitude
from librosa.feature import melspectrogram


def get_spectrogram(path):
    """Строим спектограмму из wav файла"""
    y, sr = load(path)
    S = melspectrogram(y, sr=sr, n_mels=100)
    log_S = logamplitude(S, ref_power=np.max)
    return log_S
