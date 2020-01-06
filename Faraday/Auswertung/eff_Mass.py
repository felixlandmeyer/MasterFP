import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants as cont
import uncertainties as unc
from uncertainties import ufloat
from uncertainties import unumpy as unp

def Messung(path, probe_lenth):
    lam, phi_1, min_1, phi_2, min_2 = np.genfromtxt(path, skip_header = 1, unpack =True)
    min_1 /= 60
    min_2 /= 60
    phi_1 += min_1
    phi_2 += min_2
    phi_1 /= probe_lenth #mm
    phi_2 /= probe_lenth #mm
    phi = abs(phi_1 - phi_2) / 2
    phi = np.radians(phi)
    return phi,lam

lam = Messung("data/Hochrein.txt", 5.11)[1]

P_1_phi = Messung("data/Probe_1.txt", 1.36)[0]  - Messung("data/Hochrein.txt", 5.11)[0]
P_2_phi = Messung("data/Probe_2.txt", 1.296)[0] - Messung("data/Hochrein.txt", 5.11)[0]

def fit(x,A,B):
    return A * x**2 + B

params_1, cov_1 = curve_fit(fit, lam, P_1_phi)
errors_1 = np.sqrt(np.diag(cov_1))
params_2, cov_2 = curve_fit(fit, lam, P_2_phi)
errors_2 = np.sqrt(np.diag(cov_2))

print(f"Parameter Probe 1: \n A={params_1[0]} +- {errors_1[0]} \n B = {params_1[1]} +- {errors_1[1]}")
print(f"Parameter Probe 2: \n A={params_2[0]} +- {errors_2[0]} \n B = {params_2[1]} +- {errors_2[1]}")

A_1 = ufloat(params_1[0], errors_1[0])
A_2 = ufloat(params_2[0], errors_2[0])

n = 3.3 #http://www.ioffe.ru/SVA/NSM/Semicond/GaAs/optic.html

N_1 = 1.2e18 * 1e-3 # in kubikmillimeter
N_2 = 2.8e18 * 1e-3 # in kubikmillimeter

m_1 = unp.sqrt( cont.e**3 * N_1 * 224e-3 / (8 * np.pi**2 * cont.c**3 * cont.epsilon_0 * n * A_1) )
m_2 = unp.sqrt( cont.e**3 * N_2 * 224e-3 / (8 * np.pi**2 * cont.c**3 * cont.epsilon_0 * n * A_2) )

print(f"Masse 1 {m_1/cont.m_e}")
print(f"Masse 2 {m_2/cont.m_e}")

fig,ax = plt.subplots(figsize = (5,3))
ax.scatter(lam**2,P_1_phi,color = "hotpink", label = r"Probe 1")
ax.plot(lam**2, fit(lam,*params_1), color = "magenta", label = r"Ausgleichsgerade 1")
ax.plot(lam**2, fit(lam,*params_2), color = "blue", label = r"Ausgleichsgerade 2")
ax.scatter(lam**2,P_2_phi,color = "cornflowerblue", label = r"Probe 2")
ax.set_xlabel(r"$\lambda^2 / \mu m^2 $")
ax.set_ylabel(r"$\Delta \phi_{norm.} / \frac{rad}{mm}$")
ax.legend(loc = "best")
plt.show()
