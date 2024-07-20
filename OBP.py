import math
import librosa
import numpy as np

from scipy.signal import find_peaks
from numpy import argmax, diff, nonzero

def xor_based_corr(signal):
    """
    Estimate autocorrelation via XOR
    """
    length = signal.shape[-1]
    corr = [sum([signal[j] ^ signal[(i + j) % length] for j in range(length)]) / length for i in range(length // 2)]
    return corr

def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    """
    if int(x) != x:
        raise ValueError('x must be an integer sample index')
    else:
        x = int(x)
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def freq_from_obp(sig, sr):
    """
    Estimate frequency using autocorrelation
    """
    # Calculate autocorrelation and throw away the negative lags
    corr = xor_based_corr(sig)

    # Find the first low point
    d = diff(corr)
    start = nonzero(d > 0)[0][0]

    # Find the next peak after the low point (other than 0 lag).
    # Should use a weighting function to de-emphasize the peaks at longer lags.   
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return sr / px

def f0_predictor(audio, sr, win_length=1024, hop_length=256):
    """
    Predict F0 value
    """
    sigments = librosa.util.frame(audio, frame_length=win_length, hop_length=hop_length)
    f0 = np.array([freq_from_obp(sigments[:,i], sr) for i in range(0, sigments.shape[-1])])
    return f0

def extract_sign(audio):
    """
    Extract SIGN from a signal
    """
    sign = []
    for i in range(audio.shape[-1]):
        if audio[i] >= 0:
            sign.append(1)
        else:
            sign.append(-1)
    return np.array(sign, dtype=np.int8)

audio, sr = librosa.load("/mnt/petrelfs/wangyuancheng/OneBitPitch/test/square_440.wav", sr=None)
sign = extract_sign(audio)
print(f0_predictor(sign, sr))