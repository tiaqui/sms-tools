import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import sys, os, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'/home/tiaqui/Desktop/sms-tools/software/models'))
import utilFunctions as UF

M = 501
hM1 = int(math.floor((M+1)/2))
hM2 = int(math.floor(M/2))

(fs, x) = UF.wavread('../../sounds/soprano-E4.wav')
x1 = x[5000:5000+M]*np.hamming(M)

N = 1024
# Windowing
fftbuffer = np.zeros(N)
# Last part at the begginning
fftbuffer[:hM1] = x1[hM2:]
# First part the end
fftbuffer[N-hM2:] = x1[:hM2]

# Computing
X = fft(fftbuffer)
mX = abs(X)
pX = np.unwrap(np.angle(X))
mXdb = 20*np.log10(mX)
