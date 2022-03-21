# By Liam Boutin

# Purpose of program is to read in data, apply a Hann window, compute and plot a Fourier transform of the data as well as the inverse fast Fourier transform
# Program written as part of the course requirements for ENCH 519, taken Winter 2021

from scipy import fftpack
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

#Read in data and apply Hann window
df = pd.read_csv(r'C:\Users\bouti\Desktop\Data - Quick Access\mark_relay_feedback_padded.csv', delimiter=",", names=["time", "output"])

output = df.output.values
hann_window = signal.hann(len(output))
hann_output = output * hann_window #Apply Hann window

#Plot the original data and the windowed data
fig, ax = plt.subplots()
plt.title('Controller Output vs. Time')
ax.plot(df.index, output, label = "Original")
ax.plot(df.index, hann_output, label = "Hann Window Applied")
ax.set_ylabel("Output (arb. units)")
ax.set_xlabel("Time Index")
ax.legend(loc=0)
plt.show()

#compute and plot the Fourier transform of the windowed data
hann_output_fft = fftpack.fft(hann_output)
f = fftpack.fftfreq(4096, 0.01) #Calculate the frequencies corresponding to each bin
FilterVal = 0.1 #Low pass filter value
mask = f < FilterVal
hann_output_fft_filtered = hann_output_fft*(abs(f) < FilterVal)

fig, ax = plt.subplots()
plt.title('Fourier Magn. vs. Fourier Freq. (LPF at f = ' +str(FilterVal)+' Hz)')
ax.plot(f[mask], abs(hann_output_fft[mask]), lw=2)
ax.set_ylabel("Magnitude (arb. units)")
ax.set_xlabel("Frequency (Hz)")
plt.grid(which='both', axis='both')
plt.xlim([0, 3])
plt.show()

#Plotting unfiltered data
#fig, ax = plt.subplots()
#plt.title('Fourier Magn. vs. Fourier Freq. (Unfiltered)')
#ax.plot(f, np.abs(HannOutputFFT), lw=2)
#ax.set_ylabel("Magnitude (arb. units)")
#ax.set_xlabel("Frequency (Hz)")
#plt.grid(which='both', axis='both')
#plt.xlim([0, 3])

#Compute and plot the inverse FFT
smooth_output = fftpack.ifft(hann_output_fft_filtered)

fig, ax = plt.subplots()
plt.title('Smoothed Output and Original Output vs Time (LPF at f = ' +str(FilterVal)+' Hz)')
ax.plot(df.index, output, label = "Original Data")
ax.plot(df.index, smooth_output, label = "Smoothed (Transformed) Data")
#ax.plot(df.index, Output, label = "Original")

ax.set_ylabel("Output (arb. units)")
ax.set_xlabel("Time Index")
plt.xlim([0.0001, 4000])
ax.legend(loc=0)
plt.show()


