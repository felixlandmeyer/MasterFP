import numpy as np
import pandas as pd
import uncertainties as unc
import matplotlib as mpl
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

#plt.style.use('dark_background')

Winkel, I_max, I_min, Kontrast = np.genfromtxt("data/Kontrast.txt", skip_header = 1 ,unpack = True )
Kontrast = (I_max - I_min)/(I_max + I_min)

def f(phi,A):
    return A *  abs(np.sin(2*np.radians(phi)))
phi = np.linspace(min(Winkel), max(Winkel), 1000)


params, cov = curve_fit(f, Winkel, Kontrast)
errors = np.sqrt(np.diag(cov))


pol_winkel = phi[f(phi,*params) == max(f(phi,*params))]

print(f"Kontast aus Ausgleichsrechnung:{params[0]}+-{errors[0]}")
print(f"Polarisationswinkel bei maximalem Kontrast:{pol_winkel}")

fig, ax = plt.subplots(figsize = (5,3))
ax.scatter(Winkel,Kontrast,c = "red",label = r"Kontrast")
ax.plot(phi, f(phi,*params), c = "black", label =r"Ausgleichsrechnung")
ax.axhline(y=max(f(phi,*params)),xmin=0,xmax=180 )
ax.set_xlabel(r"Polarisationswinkel")
ax.set_ylabel(r"Kontrast")
ax.set_ylim(0,1)
ax.legend(loc = "best")
plt.savefig("Plots/Kontrast.png", bbox_inches = "tight",dpi = 300)
#plt.show()
