# Simple signal analyzer based on FFT

import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

Fs = 48000

period = 1 / Fs
tend = 3
time_array1 = np.arange(0, tend, period)
time_array2 = time_array1
f1 = 1000
f2 = 1500
# generate signal 1
signal1 = 0.5*np.sin(2*np.pi*f1*time_array1)
# generate signal 2
signal2 = 0.5*np.sin(2*np.pi*f2*time_array2)

fftsignal1 = fft(signal1,norm = 'forward')

fftsignal1_re = fftsignal1.real

fftsignal1_abs = abs(fftsignal1)

# plot figure
fig, axs = plt.subplots(2, 1)
axs[0].plot(time_array1, signal1)
axs[0].set_xlim(0, tend)
axs[0].set_xlabel('time')
axs[0].set_ylabel('signal1')
axs[0].grid(True)


freq_array2 = np.arange(0, len(fftsignal1_abs), 1)
axs[1].plot(freq_array2, fftsignal1_abs)
axs[1].set_xlim(0, Fs/2)
axs[1].set_xlabel('frequency,Hz')
axs[1].set_ylabel('signal1')
axs[1].set_yscale('log')
axs[1].grid(True)
fig.tight_layout()
plt.show()