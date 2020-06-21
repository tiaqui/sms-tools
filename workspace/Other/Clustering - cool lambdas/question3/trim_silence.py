#!/usr/bin/env python3
import os
import sys
import numpy as np
from scipy.signal import get_window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import stft
import utilFunctions as UF

eps = np.finfo(float).eps

def trimSilence(inputWavPath, outputWavPath, window='hamming', M=1001, N=1024, H=256, threshold=-50):
    """Trims frames in the input WAV having an energy < threshold. 

    Inputs:
            inputWavPath (string): input sound file (monophonic with sampling rate of 44100)
            outputWavPath (string): path to the WAV file with the silence removed.
            window (string): analysis window type (choice of rectangular, triangular, hanning, 
                hamming, blackman, blackmanharris)
            M (integer): analysis window size (odd positive integer)
            N (integer): FFT size (power of 2, such that N > M)
            H (integer): hop size for the stft computation
    """
    w = get_window(window, M, False)
    fs, x = UF.wavread(inputWavPath)
    xmX, xpX = stft.stftAnal(x, w, N, H)
    db2l = lambda db: 10 ** (db/20)
    l2db = lambda l: 10 * np.log10(l)
    calcE = lambda a: sum(a**2)
    E = np.apply_along_axis(lambda a: l2db(calcE(db2l(a))), 1, xmX)
    #tAxis = np.arange(len(E)) * (H / fs)
    #plt.plot(tAxis, E)
    xTrimmed = []
    for i in range(len(E)):
        if E[i] > threshold:
            xTrimmed.extend(x[i*H:(i+1)*H])
    UF.wavwrite(np.array(xTrimmed), fs, outputWavPath)

if __name__ == '__main__':
    inputWavPath = sys.argv[1]
    outputWavPath = os.path.splitext(inputWavPath)[0] + '_trimmed.wav'
    trimSilence(inputWavPath, outputWavPath)
