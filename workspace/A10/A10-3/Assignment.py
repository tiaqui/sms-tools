import os
import sys
import numpy as np
import math
from scipy.fftpack import ifft, fftshift
from scipy.signal import blackmanharris, triang
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/')) # watch out for folder location

import utilFunctions as UF
import dftModel as DFT
import sineModel as SM

def assignment():
    # Load sounds
    fs, x1 = UF.wavread('../../sounds/orchestra.wav') # watch out for folder location
    fs, x2 = UF.wavread('db_vol1.wav')
    # Do analysis/synthesis with modified model
    ya = sineModel_multi(x1, fs, w=[get_window('hamming', (2**12)-1), get_window('blackman', (2**10)-1),
        get_window('blackman', (2**8)-1)], B=[990, 5600]) # a = [990, 5600]
    yb = sineModel_multi(x2, fs, w=[get_window('hamming', (2**12)-1), get_window('blackman', (2**10)-1),
        get_window('blackman', (2**8)-1)], B=[2500, 13000])
    # Write the output
    UF.wavwrite(ya, fs, 'orchestra_modOut.wav')
    UF.wavwrite(yb, fs, 'drum&bass_modOut.wav')
    # Do analysis/synthesis with regular model
    yc = SM.sineModel(x1, fs, N=2**12, w=get_window('blackman', (2**10)-1), t=-80)
    yd = SM.sineModel(x2, fs, N=2**12, w=get_window('blackman', (2**10)-1), t=-80)
    # Write the output
    UF.wavwrite(yc, fs, 'orchestra_regOut.wav')
    UF.wavwrite(yd, fs, 'drum&bass_regOut.wav')
    return

def sineModel_multi(x, fs, w=[get_window('blackman', (2**12)-1), get_window('blackman', (2**11)-1),
    get_window('hamming', (2**10)-1)], N=2**12, t=-80, B=[1000.0, 5000.0]):
    Bn = np.hstack((0.0, B, float(fs)/2))
    Ns = 512                                                # FFT size for synthesis (even)
    H = Ns//4                                               # Hop size used for analysis and synthesis
    hNs = Ns//2                                             # half of synthesis FFT size
    yw = np.zeros(Ns)                                       # initialize output sound frame
    y = np.zeros(x.size)                                    # initialize output array
    sw = np.zeros(Ns)                                       # initialize synthesis window
    ow = triang(2*H)                                        # triangular window
    bh = blackmanharris(Ns)                                 # blackmanharris window
    bh = bh / sum(bh)                                       # normalized blackmanharris window
    sw[hNs-H:hNs+H] = ow                                    # add triangular window
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window
    long_w = max(w[0].size, w[1].size, w[2].size)           # longest window size
    hM1 = int(math.floor((long_w+1)/2))                     # half analysis window size by rounding
    hM2 = int(math.floor(long_w/2))                         # half analysis window size by floor
    pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window
    pend = x.size - max(hNs, hM1)                           # last sample to start a frame
    while pin<pend:                                         # while input sound pointer is within sound
        #-----analysis-----
        for i in np.arange(3):
            hM1x = int(math.floor((w[i].size+1)/2))                  # half analysis window size by rounding
            hM2x = int(math.floor(w[i].size/2))                      # half analysis window size by floor
            w[i] = w[i] / sum(w[i])                                  # normalize analysis window
            x1 = x[pin-hM1x:pin+hM2x]                                # select frame
            mX, pX = DFT.dftAnal(x1, w[i], N)                        # compute dft
            ploc = UF.peakDetection(mX, t)                           # detect locations of peaks
            plocx = ploc[np.logical_and(Bn[i]*N/float(fs)<ploc,
                ploc<Bn[i+1]*N/float(fs))]                           # select location of peaks for corresponding band
            iplocx, ipmagx, ipphasex = UF.peakInterp(mX, pX, plocx)  # refine peak values by interpolation
            if i == 0:
                iploc, ipmag, ipphase = iplocx, ipmagx, ipphasex
            else:
                iploc, ipmag, ipphase = np.append(iploc, iplocx), np.append(ipmag, ipmagx), np.append(ipphase, ipphasex)
        #-----synthesis-----
        ipfreq = fs*iploc/float(N)                                   # convert peak locations to Hertz
        Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)          # generate sines in the spectrum
        fftbuffer = np.real(ifft(Y))                                 # compute inverse FFT
        yw[:hNs-1] = fftbuffer[hNs+1:]                               # undo zero-phase window
        yw[hNs-1:] = fftbuffer[:hNs+1]
        y[pin-hNs:pin+hNs] += sw*yw                                  # overlap-add and apply a synthesis window
        pin += H                                                     # advance sound pointer
    return y
