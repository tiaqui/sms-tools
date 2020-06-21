from scipy.fftpack import fft
import numpy as np
import loadTestCases

testcase = loadTestCases.load(2, 2)
[x, fs, f] = testcase['input']


M = len(x)
T = fs/f
zpn = int(T*(float(M)/T - np.floor(M/T)))  # zero padding neccesary
N = M + zpn
hM1 = int(np.floor((M+1)/2))
hM2 = int(np.floor(M/2))
fftbuffer = np.zeros(N)
fftbuffer[:hM1] = x[hM2:]
fftbuffer[N-hM2:] = x[:hM2]
X = fft(fftbuffer)
mX = 20*np.log10(abs(X[:int(N/2+1)]))
