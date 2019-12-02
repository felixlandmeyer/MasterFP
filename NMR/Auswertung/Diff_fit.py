import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl


def theory(tau, M0, M1, T):
    T2=560
    return M0 * np.exp(-2 * tau / T2) * np.exp(-tau**3 / T) + M1

tau, U = np.genfromtxt('../data/Diffusion.txt', unpack=True)
tau *=1e3


params, cov = curve_fit(theory, tau, U) #,p0=[])
errors = np.sqrt(np.diag(cov))
print('M0 = ', params[0], '±', errors[0])
print('M1 = ', params[1], '±', errors[1])
print('T =', params[2], '±', errors[2])

x = np.linspace(0, 32, 1000)
plt.plot(tau, U, 'x',label='Messwerte', color='indianred', linewidth=0.5)
plt.plot(x, theory(x, *params), label='Fit', color='dodgerblue')
# plt.fill_between(x, theory(x, *params)+theory(x, *errors), theory(x, *params)-theory(x, *errors), alpha = 0.4, color = 'dodgerblue')
plt.xlabel(r'$t\,\, / \,\, \mathrm{ms}$')
plt.ylabel(r'$Signalstärke \,\, / \,\, \mathrm{mV}$')
plt.legend(loc='best')
#plt.show()
plt.savefig('Diff_fit.png')



