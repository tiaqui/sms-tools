import numpy as np
import matplotlib.pyplot as plt

N = 64
k0 = 7

# Exponential sine
#x = np.exp(1j*2*np.pi*k0*np.arange(N)/N)

# Real sine
x = np.cos(2*np.pi*k0*np.arange(N)/N)
nv = np.arange(-N/2, N/2)
kv = np.arange(-N/2, N/2)

X = np.array([])

for k in range(N):
    s = np.exp(1j*2*np.pi*k*nv/N)
    X = np.append(X, sum(x*np.conjugate(s)))

# Exp
#plt.plot(np.arange(N), abs(X))
#plt.axis([0, N-1, 0, N])

# Real
plt.plot(kv, abs(X))
plt.axis([-N/2, (N/2)-1, 0, N])

plt.show()
