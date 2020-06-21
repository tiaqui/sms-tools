import numpy as np
import matplotlib.pyplot as plt
import loadTestCases
# import A3Part1
"""
# A3Part1
testcase = loadTestCases.load(1, 1)
output = A3Part1.minimizeEnergySpreadDFT(**testcase['input'])
"""

# A3Part2
# Gen sine
"""
N = 25.0
f = 100.0
fs = 1000.0
A = 1

n = np.arange(N)
y = A*np.cos(2*np.pi*f*n/fs)
"""
"""
import A3Part2
testcase1 = loadTestCases.load(2, 1)
out1 = testcase1['output']
outmX1 = A3Part2.optimalZeropad(**testcase1['input'])
print('Case1', out1, outmX1)
testcase2 = loadTestCases.load(2, 2)
out2 = testcase2['output']
outmX2 = A3Part2.optimalZeropad(**testcase2['input'])
print('Case2', out2, outmX2)
# testcase3 = loadTestCases.load(3, 3)
# out3 = testcase3['output']

# plt.plot(outmX)
# plt.plot(out)
# plt.show()
"""
"""
# A3Part3

import A3Part3
# x = np.array([2, 3, 4, 3, 2])
# out = A3Part3.testRealEven(x)
# X = out[2]
testcase1 = loadTestCases.load(3, 1)
testcase2 = loadTestCases.load(3, 2)

o1 = testcase1['output']
o2 = testcase2['output']

out1 = A3Part3.testRealEven(**testcase1['input'])
out2 = A3Part3.testRealEven(**testcase2['input'])
print('Case1')
print(o1)
print(out1)
print('Case 2')
print(o2)
print(out2)
"""
# A3Part4
"""
import A3Part4
test1 = loadTestCases.load(4, 1)
test2 = loadTestCases.load(4, 2)

out1 = A3Part4.suppressFreqDFTmodel(**test1['input'])
out2 = A3Part4.suppressFreqDFTmodel(**test2['input'])

o1 = test1['output']
o2 = test2['output']

print('Case 1, Deberia ser', o1, 'Es', out1)
print('Case 2, Deberia ser', o2, 'Es', out2)

# XI = np.imag(X)
# dm = 0
# for i in np.arange(len(XI)-1):
#    d = abs(XI[i]-XI[i+1])
#    if d > dm:
#        dm = d
# if dm < 1e-6:
#    isRealEven = 'true'
"""
# A3Part5
import A3Part5
"""
test = loadTestCases.load(5, 1)

o = test['output']
out = A3Part5.zpFFTsizeExpt(**test['input'])

o1 = o[0]
o2 = o[1]
o3 = o[2]

out1 = out[0]
out2 = out[1]
out3 = out[2]

plt.plot(o1)
plt.plot(o2)
plt.plot(o3)
plt.show()

print(max(o1), max(o2), max(o3))
"""
# Sine
N = 512
f = 100.0
fs = 1000.0

n = np.arange(N)
y = np.cos(2*np.pi*f*n/fs)

Y = A3Part5.zpFFTsizeExpt(y, fs)

Y1 = Y[0]
Y2 = Y[1]
Y3 = Y[2]

plt.plot(Y1)
plt.plot(Y2)
plt.plot(Y3)
plt.show()
