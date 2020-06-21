from A2Part1 import genSine
#from A2Part2 import genComplexSine
#from A2Part3 import DFT
from A2Part4 import IDFT
#from A2Part5 import genMagSpec

import matplotlib.pyplot as plt
import numpy as np

# Part1

A = 1.0
phi = 1.0
fs = 40.0
t = 1.0
f = 25.0

y = genSine(A, f, phi, fs, t)
time = np.arange(0, t, 1/fs)

plt.plot(time, y)
plt.show()

# Part2
"""
N = 5.0
k = 1.0

y = genComplexSine(k, N)

print(y)
"""
# Part 3
"""
x = np.array([1, 2, 3, 4])
X = DFT(x)

print(X)
"""
# Part 4
X = np.array([1, 1, 1, 1])

x = IDFT(X)

print(x)
plt.plot(np.arange(len(x)), abs(x))
# Part 5
"""
x = np.array([1, 2, 3, 4])
mag = genMagSpec(x)

print(mag)
"""
