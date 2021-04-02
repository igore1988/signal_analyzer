# Simple signal analyzer based on FFT

import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

Fs = 48000

period = 1 / Fs
tend = 3
time_array1 = np.arange(0, tend, period)
time_array2 = time_array1
f1 = 10
f2 = 15
# generate signal 1
signal1 = 0.5*np.sin(2*np.pi*f1*time_array1)
# generate signal 2
signal2 = 0.5*np.sin(2*np.pi*f2*time_array2)

fftsignal1 = fft(signal1)

# plot figure
fig, axs = plt.subplots(2, 1)
axs[0].plot(time_array1, signal1)
axs[0].set_xlim(0, tend)
axs[0].set_xlabel('time')
axs[0].set_ylabel('signal1')
axs[0].grid(True)

axs[1].plot(time_array2, fftsignal1)
axs[1].set_xlim(0, 1)
axs[1].set_xlabel('time')
axs[1].set_ylabel('signal1')
axs[1].grid(True)

plt.show()