import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks_cwt


times, imag_v, real_v = np.genfromtxt('../data/scope_80.csv', unpack=True, delimiter=',')

#Suchen des Echo-Maximums und alle Daten davor abschneiden
start = np.argmax(real_v)
print('Startwert:',start)
times = times[start:]
real = real_v[start:]
imag = imag_v[start:]

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


#Echo vor Abschneiden und Phasenkorrektur
#plt.plot(times,real_v,'-',color='indianred', label='Realteil')
#plt.plot(times,imag_v, '-', color='dodgerblue', label='Imaginärteil')
#plt.xlabel(r'$t\,\, / \,\, \mathrm{s}$')
#plt.ylabel(r'$Signalstärke \,\, / \,\, \mathrm{V}$')
#plt.legend(loc='best')
#plt.savefig("echo_vorher.pdf")

#Echo nach Abschneiden
#plt.plot(times,real,'-',color='indianred', label='Realteil')
#plt.plot(times,imag, '-', color='dodgerblue', label='Imaginärteil')
#plt.xlabel(r'$t\,\, / \,\, \mathrm{s}$')
#plt.ylabel(r'$Signalstärke \,\, / \,\, \mathrm{V}$')
#plt.legend(loc='best')
#plt.savefig("echo_abgeschnitten.pdf")

test=freqs/1000
test_x=[test[262],test[274]]
test_y=[np.real(fftdata)[262],np.real(fftdata)[274]]
print('Frequenzverschiebung =', test[274]-test[262], 'kHz')

plt.plot((freqs/1000), np.real(fftdata), 'x', color='grey', label='Werte der Fouriertrafo')
plt.plot(test_x,test_y, 'rx')
plt.xlim(-20, 20)
plt.xlabel(r'$Frequenzverschiebung \,\, / \,\, \mathrm{kHz}$')
plt.ylabel(r'$Intensität$')
plt.legend(loc='upper left', prop={'size': 9})
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