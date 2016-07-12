__author__ = 'Kirill Rudakov'

import numpy as np
import scipy.signal as signal
from scipy import interpolate
import matplotlib.pyplot as plt
import peakutils
import time

# -- Initialize interpolation, low and high frequency variables in Hz
INTERPOLATION_FREQ = 4
LF_MIN = 0.04
LF_MAX = 0.15
HF_MIN = 0.15
HF_MAX = 0.4

# get spectrum power
def get_spectrum_power(spectrum, freq, fmin, fmax):
    span = [spectrum[i] for i in range(len(spectrum)) if freq[i] >= fmin and freq[i]<fmax]
    return np.sum(span)

# get TP, LF, HF and LF/HF
def get_frequences(rr,time_axis):
    # -- Normalize non-normal peaks
    rr_peak_index = peakutils.indexes(rr,thres=0.1, min_dist=5) # get non-normal peaks
    time_nn, rr_nn = [], []

    # find non-normal's neighbours
    for i in range(len(rr)):
        if i in rr_peak_index:
            # try-except for boundary instances
            try:
                time_nn.append(time_axis[i-1])
                rr_nn.append(rr[i-1])
            except:
                pass
            try:
                time_nn.append(time_axis[i+1])
                rr_nn.append(rr[i+1])
            except:
                pass

    # interpolate non-normal peaks
    interpolate_nn = interpolate.interp1d(time_nn,rr_nn)
    nn_rr = interpolate_nn([element for i,element in enumerate(time_axis) if i in rr_peak_index])
    for i in range(len(nn_rr)):
        rr[rr_peak_index[i]] = nn_rr[i]

    # -- Prepare RR
    time_grid = np.arange(np.min(time_axis),np.max(time_axis),0.25) # time grid with freq step 1/INTERPOLATION_FREQ

    # interpolate data
    interpolate_rr = interpolate.interp1d(time_axis, rr, kind='cubic')
    interpolated_rr = interpolate_rr(time_grid)

    # remove linear trend along axis from data
    detrend_rr = interpolated_rr - np.mean(interpolated_rr)

    # use Tukey window for smoothing curves
    # window = signal.tukey(len(detrend_rr),alpha=0.25)
    import tukey
    window = tukey.tukey(len(detrend_rr),alpha=0.25)
    detrend_rr = (window*detrend_rr)/1000

    # -- Obtaing spectrum
    spectr = (np.absolute(np.fft.fft(detrend_rr,2048))) # zero padding to 2 ^ 11
    spectr = spectr[0:len(spectr)/2] # use only positive section

    freqs = np.linspace(start=0, stop=INTERPOLATION_FREQ/2, num=len(spectr), endpoint=True) # frequence space

    # plot
    # plt.plot(freqs[:len(spectr)/4],spectr[:len(spectr)/4])
    # plt.show()

    # -- Return TP, LF, HF and LF/HF
    LF = get_spectrum_power(spectr, freqs, LF_MIN, LF_MAX)
    HF = get_spectrum_power(spectr, freqs, HF_MIN, HF_MAX)
    TP = get_spectrum_power(spectr, freqs, 0., HF_MAX)

    return(TP,LF,HF,LF/HF)

def get_time_and_RR(file):
    data = open(file,'rb')
    # raw - 1st column is RR, 2nd is HR
    try:
        raw = np.genfromtxt(data, delimiter=',')
    except:
        raw = []
        for line in data:
            raw.append([float(i) for i in line.split(',')])
        raw = np.array(raw)

    rr = raw[:, 0]
    # hr = raw[:, 1]

    time_axis = np.cumsum(rr)/1000 # time in sec

    return rr,time_axis

# -- Example
# rr,time_axis = get_time_and_RR('edis_test.txt')
# print(get_frequences(rr,time_axis))