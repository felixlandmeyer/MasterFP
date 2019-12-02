import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks_cwt


times, real, imag = np.genfromtxt('../Nicole/diffusion_transient.csv', unpack=True, delimiter=',')

#Suchen des Echo-Maximums und alle Daten davor abschneiden
start = np.argmax(real)
times = times[start:]
real = real[start:]
imag = imag[start:]

#Phasenkorrektur - der Imaginärteil bei t=0 muss = 0 sein
phase = np.arctan2(imag[0], real[0])

#Daten in komplexes Array mit Phasenkorrektur speichern 
compsignal = (real*np.cos(phase)+imag*np.sin(phase))+(-real*np.sin(phase)+imag*np.cos(phase))*1j

#Offsetkorrektur, ziehe den Mittelwert der letzten 512 Punkte von allen Punkten ab
compsignal -= compsignal[-512:-1].mean()

#Der erste Punkt einer FFT muss halbiert werden
#compsignal[0] = compsignal[0]/2.0

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

#Erstellen eines Plots
plt.plot(times,real,'r-')
plt.plot(times,imag, 'b-')
plt.savefig("echo.pdf")

test=[]
print(freqs)

plt.plot(freqs, np.real(fftdata), 'rx')
plt.xlim(-10000, 10000)
plt.xlabel(r'$Frequenzverschiebung \,\, / \,\, \mathrm{Hz}$')
plt.ylabel(r'$Intensität$')
plt.legend(loc='best')
#plt.show()
plt.savefig("echo_ft.pdf")

'''
plt.plot(t, U1, 'x',label='Messwerte', color='indianred', linewidth=0.5)
plt.plot(t, U2, 'x',label='Messwerte', color='dodgerblue', linewidth=0.5)
plt.xlabel(r'$t\,\, / \,\, \mathrm{s}$')
plt.ylabel(r'$Signalstärke \,\, / \,\, \mathrm{V}$')
plt.legend(loc='best')
plt.show()
'''