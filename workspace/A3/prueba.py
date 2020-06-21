# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import triang
from scipy.fftpack import fft
from matplotlib import pyplot as plt

x = triang(15)
fft_buffer = np.zeros(len(x))
fft_buffer[:8] = x[7:]
fft_buffer[8:] = x[:7]

#X = np.fft.fft(fft_buffer)
X = fft(fft_buffer)
mX = abs(X)
pX = np.angle(X)

