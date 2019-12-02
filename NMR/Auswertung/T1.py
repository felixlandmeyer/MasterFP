import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl

def theory(tau, M0, M1, T1):
    return M0 * np.exp(-tau/T1) + M1

tau, U = np.genfromtxt('T1.txt', unpack=True)

params, cov = curve_fit(theory,tau, U, p0=[2.91,-1.33,1])
errors = np.sqrt(np.diag(cov))

print('M0 = ', params[0], '±', errors[0])
print('M1 = ', params[1], '±', errors[1])
print('T1 =', params[2], '±', errors[2])

x = np.linspace(0, 10, 1000)
plt.plot(x, theory(x, *params), label='Fit', color='dodgerblue')
plt.plot(tau, U, 'x', color='indianred', label='Messwerte')
plt.xlabel(r'$\tau\,\, / \,\, \mathrm{s}$')
plt.ylabel(r'$Signalstärke \,\, / \,\, \mathrm{V}$')
plt.xscale('log')
plt.legend(loc='best')
#plt.show()
plt.savefig('T1_fit.pdf')
