# A1 test
"""
from A1Part1 import readAudio
import sys
sys.path.append('../../software/models/')
from utilFunctions import wavread
path = '../../sounds/'
file = 'piano.wav'

ret = readAudio(path+file)
ret
"""
# A2 test
"""
from A1Part2 import minMaxAudio

path = '../../sounds/'
file = 'oboe-A4.wav'

juan = minMaxAudio(path+file)
print(juan)
"""
# A3 test
"""
import numpy as np
from A1Part3 import hopSamples

x = np.arange(10)
y = hopSamples(x, 2)
print(y)
"""
# A4 test
"""
from A1Part4 import downsampleAudio

path = '../../sounds/'
file = 'vibraphone-C6.wav'

y = downsampleAudio(path+file, 16)
"""
