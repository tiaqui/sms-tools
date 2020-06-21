import matplotlib.pyplot as plt
import numpy as np

# Complex sine
N = 500
k = 3
n = np.arange(-N/2, N/2)
s = np.exp(1j*2*np.pi*k*n/N)

plt.plot(n, np.real(s))
#plt.plot(n, np.imag(s))
plt.axis([-N/2, N/2-1, -1, 1])

plt.show()
