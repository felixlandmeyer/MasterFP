import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks_cwt
import uncertainties as unc
from uncertainties import ufloat
import scipy.constants

times_v, imag_v, real_v = np.genfromtxt('../data/scope_80.csv', unpack=True, delimiter=',')

#Suchen des Echo-Maximums und alle Daten davor abschneiden
start = np.argmin(imag_v)
print('Startindice:',start)
print('Startwert:',imag_v[start])
times = times_v[start:]
real = real_v[start:]
imag = imag_v[start:]

#Echo vor Abschneiden und Phasenkorrektur
plt.plot(times_v,real_v,'-',color='indianred', label='Realteil')
plt.plot(times_v,imag_v, '-', color='dodgerblue', label='Imaginärteil')
#plt.xlim(0,times_v[-1])
plt.xlabel(r'$t\,\, / \,\, \mathrm{s}$')
plt.ylabel(r'$Signalstärke \,\, / \,\, \mathrm{V}$')
plt.legend(loc='best')
plt.savefig("echo_vorher.pdf", bbox_inches = "tight")
plt.clf()

#Echo nach Abschneiden
plt.plot(times,real,'-',color='indianred', label='Realteil')
plt.plot(times,imag, '-', color='dodgerblue', label='Imaginärteil')
plt.xlabel(r'$t\,\, / \,\, \mathrm{s}$')
plt.ylabel(r'$Signalstärke \,\, / \,\, \mathrm{V}$')
plt.legend(loc='best')
plt.savefig("echo_abgeschnitten.pdf", bbox_inches = "tight")
plt.clf()

#Phasenkorrektur - der Imaginärteil bei t=0 muss = 0 sein
phase = np.arctan2(imag[0], real[0])

#Daten in komplexes Array mit Phasenkorrektur speichern
compsignal = (real*np.cos(phase)+imag*np.sin(phase))+(-real*np.sin(phase)+imag*np.cos(phase))*1j

#Offsetkorrektur, ziehe den Mittelwert der letzten 512 Punkte von allen Punkten ab
compsignal -= compsignal[-512:-1].mean()

#Der erste Punkt einer FFT muss halbiert werden
compsignal[0] = compsignal[0]/2.0

#Anwenden einer Fensterfunktion (siehe z. Bsp. #https://de.wikipedia.org/wiki/Fensterfunktion )
#Hier wird eine Gaußfunktion mit sigma = 100 Hz verwendet
apodisation = 100.0*2*np.pi
compsignal = compsignal*np.exp(-1.0/2.0*((times-times[0])*apodisation)**2)

#Durchführen der Fourier-Transformation
fftdata = np.fft.fftshift(np.fft.fft(compsignal))

#Generieren der Frequenzachse
freqs = np.fft.fftshift(np.fft.fftfreq(len(compsignal), times[1]-times[0]))

#Speichern des Ergebnisses als txt
np.savetxt("echo_gradient_fft.txt", np.array([freqs, np.real(fftdata), np.imag(fftdata)]).transpose())



test=freqs/1000
shoulder_low = 283
shoulder_high = 292
test_x=[test[shoulder_low],test[shoulder_high]]
test_y=[np.real(fftdata)[shoulder_low],np.real(fftdata)[shoulder_high]]

d_f = (test[shoulder_high]-test[shoulder_low]) * 10**3

gamma_proton = 42.577 * 10**6
k = scipy.constants.k
eta = 2.95e-3
T = (21.5 + 273.15)
T_D = ufloat(5.57e-6,0.25e-6)
d = 4.2e-3

Grad = 2 * np.pi * d_f / (d * gamma_proton)

D = 3 / (2 * T_D * gamma_proton**2 * Grad**2)

R = k * T / (6 * np.pi * eta * D)

print('Frequenzverschiebung =', d_f, 'kHz')
print(f"Gradient = {Grad} T/m")
print(f"Diffusionskonstante= {D} in m^2/s")
print(f"Radius= {R} in m")



#Plott zur Frequenz
plt.plot((freqs/1000), np.real(fftdata), 'x', color='grey', label='Werte der Fouriertrafo')
plt.plot(test_x,test_y, 'rx', color = "red")
plt.xlim(-20, 20)
plt.xlabel(r'$Frequenzverschiebung \,\, / \,\, \mathrm{kHz}$')
plt.ylabel(r'$Intensität$')
plt.legend(loc='upper left', prop={'size': 9})
#plt.show()
plt.savefig("echo_ft.pdf", bbox_inches = "tight")
plt.clf()

#Plott nach Phasenkorrektur
plt.plot(times, np.imag(compsignal), '-',label='Realteil',  color='indianred')
plt.plot(times, -np.real(compsignal), '-',label='Imaginärteil',color='dodgerblue')
plt.xlabel(r'$t\,\, / \,\, \mathrm{s}$')
plt.ylabel(r'$Signalstärke \,\, / \,\, \mathrm{V}$')
plt.legend(loc='best')
plt.savefig("echo_phasenkorrektur.pdf", bbox_inches = "tight")
