import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks_cwt


t, U = np.genfromtxt('../Nicole/T2_MG.csv', unpack=True, delimiter=',')

peaks = find_peaks_cwt(U, np.arange(1,400))
tp, Up = t[peaks], U[peaks]
tp=tp[6:107]
Up=Up[6:107]


def theory(t, M0, M1, T2):
    return M0 * np.exp(-t/T2) + M1

params, cov = curve_fit(theory, tp, Up)
errors = np.sqrt(np.diag(cov))
print('M0 = ', params[0], '±', errors[0])
print('M1 = ', params[1], '±', errors[1])
print('T2 =', params[2], '±', errors[2])

x = np.linspace(-0.1, 1.85, 100)
plt.plot(t, U, '-',label='Messwerte', color='indianred', linewidth=0.5)
plt.plot(tp,Up,'x', label='Maxima', color='dodgerblue')
plt.plot(x, theory(x, *params), label='Fit', color='black')
plt.xlabel(r'$t\,\, / \,\, \mathrm{s}$')
plt.ylabel(r'$Signalstärke \,\, / \,\, \mathrm{V}$')
plt.legend(loc='best')
#plt.show()
plt.savefig('T2_fit.pdf')



