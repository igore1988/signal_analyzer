# Simple signal analyzer based on FFT

import numpy as np
from scipy.fft import fft, ifft, rfft, irfft, fftshift, ifftshift
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.signal import convolve
import scipy.signal.windows as win

Fs = 48000
period = 1 / Fs
tend = 1
time_array1 = np.arange(0, tend, period)
time_array2 = time_array1
f1 = 1000
f2 = 15000
# generate signal 1
signal1 = 0.5*np.sin(2*np.pi*f1*time_array1)
# generate signal 2
signal2 = 0.5*np.sin(2*np.pi*f2*time_array2)
# generate signal 3
signal3 = 0.25*np.exp(1j*np.pi*f2*time_array2)
# create impulse response of low-pass FIR filter
fir_order = 64
# allocate memory for FIR filter
fir_number = np.arange(0, fir_order, 1)
fir_array = np.zeros(fir_order, dtype=complex)
fir_array[0:int(fir_order/8)] = 1.0
fir_array_comp = fir_array + 1j*fir_array
imp_fir_response = ifftshift(ifft(fir_array_comp))
# convolution
signal_sum = signal1 + signal2
window = win.chebwin(64, at=60)
imp_fir_response = np.multiply(imp_fir_response, imp_fir_response)
signal_filtered = convolve(signal_sum, imp_fir_response, mode='same')



fftsignal_sum = fft(signal_sum)
fftsignal_filt = fft(signal_filtered)
fftsignal_filt_re = fftsignal_filt.real
fftsignal_sum_pow = 10*np.log10(abs(fftsignal_sum))
fftsignal_filt_pow = 10*np.log10(abs(fftsignal_filt))

# plot figure
fig, axs = plt.subplots(3, 1)
axs[0].plot(time_array1, signal_sum.real)
axs[0].plot(time_array1, signal_filtered.real)
axs[0].set_xlim(0, tend)
axs[0].set_xlabel('Samples')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

freq_array2 = np.arange(0, len(fftsignal_filt_pow), 1)
axs[1].plot(freq_array2, fftsignal_sum_pow)
axs[1].plot(freq_array2, fftsignal_filt_pow)
axs[1].set_xlim(0, Fs/2)
axs[1].set_xlabel('frequency,Hz')
axs[1].set_ylabel('Amplitude, dB')
axs[1].set_yscale('linear')
axs[1].grid(True)

axs[2].plot(fir_number, abs(imp_fir_response))
axs[2].set_xlim(0, fir_order)
axs[2].set_xlabel('Samples')
axs[2].set_ylabel('Amplitude')
axs[2].set_yscale('linear')
axs[2].grid(True)

fig.tight_layout()
plt.show()