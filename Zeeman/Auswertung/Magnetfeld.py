import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl

I,B = np.genfromtxt("Magnetfeld.txt", skip_header=2,unpack = True)

def fit(I,a,b):
    return a * I + b

params, cov = curve_fit(fit,I, B) #, p0=[2.91,-1.33,1])
errors = np.sqrt(np.diag(cov))

print('a = ', params[0], '±', errors[0])
print('b = ', params[1], '±', errors[1])

plt.plot(I, fit(I, *params), label='Ausgleichgerade', color='magenta')
plt.plot(I, B, 'x', color='grey', label='Messwerte')
plt.xlabel(r'$Stromstärke\,\, / \,\, \mathrm{A}$')
plt.ylabel(r'$Magnetfeld \,\, / \,\, \mathrm{mT}$')
#plt.xscale('log')
plt.legend(loc='best')
#plt.show()
plt.savefig('Magnetfeld.pdf')