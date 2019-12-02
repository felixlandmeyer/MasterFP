import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks_cwt


t, U_real, U_imag = np.genfromtxt('../data/scope_75.csv', unpack=True, delimiter=',')
U_real *=1e3

peaks = find_peaks_cwt(U_real, np.arange(1,25))
tp, Up = t[peaks], U_real[peaks]
tp=tp[17:70:2]
Up=Up[17:70:2]


def theory(t, M0, M1, T2):
    return M0 * np.exp(-t/T2) + M1

params, cov = curve_fit(theory, tp, Up, p0=[1000,0,0.5])
errors = np.sqrt(np.diag(cov))
print('M0 = ', params[0], '±', errors[0])
print('M1 = ', params[1], '±', errors[1])
print('T2 =', params[2], '±', errors[2])

x = np.linspace(4.6, 5.7, 1000)
plt.plot(t, U_real, '-',label='Messwerte', color='grey', linewidth=0.5)
plt.plot(tp,Up,'x', label='Maxima', color='dodgerblue')
plt.plot(x, theory(x, *params), label='Fit', color='indianred')
plt.xlim(4.55,5.75)
plt.xlabel(r'$t\,\, / \,\, \mathrm{ms}$')
plt.ylabel(r'$Signalstärke \,\, / \,\, \mathrm{mV}$')
plt.legend(loc='best')
#plt.show()
plt.savefig('T2_fit.pdf')



