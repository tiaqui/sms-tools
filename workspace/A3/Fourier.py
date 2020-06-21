import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import triang
from scipy.fftpack import fft

x = triang(15)
# Windowing
fftbuffer = np.zeros(15)
# Last part at the begginning
fftbuffer[:8] = x[7:]
# First part at the end
fftbuffer[8:] = x[:7]

# Computing
X = fft(fftbuffer)
mX = abs(X)
pX = np.angle(X)
