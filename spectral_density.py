import matplotlib.pyplot as plt
import numpy as np

t = np.arange(3,12,1e-5)
V = np.random.normal(0,1,t.shape)

# t,V = np.loadtxt("input_file.txt",unpack=True)     # use this if you are importing data from a file

# spectral density calculation
T = t[-1] - t[0]
dt = t[1] - t[0]
sampling_rate = 1/dt

V_T = V/np.sqrt(T)
V_f = np.fft.fft(V_T)*dt
S_V = np.abs(V_f)**2
f = np.fft.fftfreq(len(V_T),dt)

W_V = 2*S_V[f>0]   # keep only positive frequencies
f = f[f>0]         # keep only positive frequencies
df = f[1]-f[0]

V_rms_time = np.sqrt(np.trapz(V**2,x=t,dx=dt)/T)
print(V_rms_time)

V_rms_freq = np.sqrt(np.trapz(W_V,x=f,dx=df))
print(V_rms_freq)

fig, ax = plt.subplots()
ax.loglog(f,np.sqrt(W_V))
ax.set_ylabel("voltage spectrum, $\sqrt{W_V}$ [V/$\sqrt{\mathrm{Hz}}$]")
ax.set_xlabel("frequency, $f$ [Hz]")
ax.margins(0,0.1)
plt.show()