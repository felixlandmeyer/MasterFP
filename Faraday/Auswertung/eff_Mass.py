import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants as cont
import uncertainties as unc
from uncertainties import ufloat
from uncertainties import unumpy as unp
import os

mpl.rcParams.update({'font.family': 'serif',
                    'text.usetex': True,
                    'pgf.rcfonts': False,
                    'pgf.texsystem': 'lualatex',
                    'pgf.preamble': r'\usepackage{unicode-math} \usepackage{siunitx}',})
#font = {'family': 'normal',
#        'weight': 'normal',
#        'size': 12}
#
#plt.rc('font', **font)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#
#os.system('mkdir -p Plots')

def Messung(path, probe_length): #probe_lentgh in meter
    lam, phi_1, min_1, phi_2, min_2 = np.genfromtxt(path, skip_header = 1, unpack =True)
    min_1 /= 60
    min_2 /= 60
    phi_1 += min_1
    phi_2 += min_2
    phi = abs(phi_1 - phi_2) / 2
    phi = np.deg2rad(phi)
    phi /= probe_length
    return phi,lam

n = 3.3 #http://www.ioffe.ru/SVA/NSM/Semicond/GaAs/optic.html

N_1 = 1.2e18*10**6 # in 1/kubikmeter
N_2 = 2.8e18*10**6 # in 1/kubikmeter

B = ufloat(np.mean([222,219]), np.std([222,219])/np.sqrt(2))
print(f"Magnetfeldstärke = {B}")

probe_length_rein = 5.11e-3 #meter
probe_length_dotiert1 = 1.36e-3 #meter
probe_length_dotiert2 = 1.296e-3 #meter

#Messung("data/Probe_2.txt", 1.296)

lam = Messung("data/Hochrein.txt", probe_length_rein)[1] #in mu
lam = lam * 10**-6 # in meter

Phi_probe_rein = Messung("data/Hochrein.txt", probe_length_rein)[0]
Phi_probe_1 = Messung("data/Probe_1.txt", probe_length_dotiert1)[0]
Phi_probe_2 = Messung("data/Probe_2.txt", probe_length_dotiert2)[0]



delta_phi_1 = Phi_probe_1 - Phi_probe_rein
delta_phi_2 = Phi_probe_2 - Phi_probe_rein

def lin(x, A, B):
    return A * x**2 + B

params_delta_phi_1, cov = curve_fit(lin, lam, delta_phi_1, p0 = [10e13,2])
errors_delta_phi_1 = np.sqrt(np.diag(cov))

A_delta_phi_1 = ufloat(params_delta_phi_1[0],errors_delta_phi_1[0])
B_delta_phi_1 = ufloat(params_delta_phi_1[1],errors_delta_phi_1[1])

params_delta_phi_2, cov = curve_fit(lin, lam, delta_phi_2,p0 = [10e13,2])
errors_delta_phi_2 = np.sqrt(np.diag(cov))

A_delta_phi_2 = ufloat(params_delta_phi_2[0],errors_delta_phi_2[0])
B_delta_phi_2 = ufloat(params_delta_phi_2[1],errors_delta_phi_2[1])

eff_m_1 = unp.sqrt( cont.e**3 * N_1 * B*10**-3 / (8 * np.pi**2 * cont.c**3 * cont.epsilon_0 * n * A_delta_phi_1) )
eff_m_2 = unp.sqrt( cont.e**3 * N_2 * B*10**-3 / (8 * np.pi**2 * cont.c**3 * cont.epsilon_0 * n * A_delta_phi_2) )

V_eff_1 = eff_m_1 / cont.m_e
V_eff_2 = eff_m_2 / cont.m_e

print(f"normierte Winkeldifferenz Hochrein:\n {Phi_probe_rein * 10**-3} in rad/mm")
print(f"normierte Winkeldifferenz Probe 1:\n {Phi_probe_1 * 10**-3}in rad/mm")
print(f"normierte Winkeldifferenz Probe 2:\n {Phi_probe_2 * 10**-3}in rad/mm")

print(f"Probe 1 \n A = {A_delta_phi_1 * 10 **-9} in rad/mm3 \n B = {B_delta_phi_1 * 10**-3} in rad/mm \n m_eff = {eff_m_1} in kg \n V_eff = {V_eff_1}")
print(f"Probe 2 \n A = {A_delta_phi_2 * 10 **-9} in rad/mm3 \n B = {B_delta_phi_2 * 10**-3} in rad/mm \n m_eff = {eff_m_2} in kg \n V_eff = {V_eff_2}")

# Plotte alle Messwerte
#fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize = (5,5))
#ax1.scatter((lam*10**6)**2, Phi_probe_rein * 10 **-3, color = "red", label = r"\phi_\text{norm. rein}")
#ax1.set_ylabel(r"$\phi_\text{norm. rein} \, / \, \si{\radian\per\milli\meter}$")
#ax2.scatter((lam*10**6)**2, Phi_probe_1 * 10 **-3, color = "blue", label = r"\phi_\text{norm. 1}")
#ax2.set_ylabel(r"$\phi_\text{norm. 1} \, / \, \si{\radian\per\milli\meter}$")
#ax3.scatter((lam*10**6)**2, Phi_probe_2 * 10 **-3, color = "green", label = r"\phi_\text{norm. 2}")
#ax3.set_ylabel(r"$\phi_\text{norm. 2} \, / \, \si{\radian\per\milli\meter}$")
#plt.tight_layout()
#plt.savefig("Plots/Messwerte_normiert.pdf", bbox_inches = "tight")
#plt.clf()
fig,ax = plt.subplots(figsize = (5,3))
plt.scatter((lam*10**6)**2, Phi_probe_rein * 10 **-3, color = "mediumvioletred", label = r"$\phi_\text{norm. rein}$")
plt.ylabel(r"$\phi_\text{norm. rein} \, / \, \si{\radian\per\milli\meter}$")
plt.legend()
plt.tight_layout()
plt.savefig("Plots/Messwerte_rein.pdf")


fig,ax = plt.subplots(figsize = (10,6))
ax.scatter((lam*10**6)**2, delta_phi_1 * 10**-3, color = "hotpink",label = r"$\Delta \phi_\text{norm.1}$")
ax.plot((lam*10**6)**2, lin(lam, *params_delta_phi_1) * 10**-3, color = "magenta", label = r"Ausgleichsgerade $\Delta\phi_\text{norm,1}$")
plt.fill_between((lam*10**6)**2,
                lin(lam, *(params_delta_phi_1+errors_delta_phi_1)) * 10**-3,
                lin(lam, *(params_delta_phi_1-errors_delta_phi_1)) * 10**-3,
                color = 'magenta', alpha = 0.15,
                )
ax.scatter((lam*10**6)**2, delta_phi_2 * 10**-3, color = "cornflowerblue",label = r"$\Delta \phi_\text{norm.2}$")
ax.plot((lam*10**6)**2, lin(lam, *params_delta_phi_2) * 10**-3 , color = "blue", label = r"Ausgleichsgerade $\Delta\phi_\text{norm,2}$")
plt.fill_between((lam*10**6)**2,
                lin(lam, *(params_delta_phi_2+errors_delta_phi_2)) * 10**-3,
                lin(lam, *(params_delta_phi_2-errors_delta_phi_2)) * 10**-3,
                color = 'blue', alpha = 0.15,
                )
ax.set_ylabel(r"$\Delta\phi_\text{norm.} \, / \, \si{\radian\per\milli\meter} $")
ax.set_xlabel(r"$\lambda^2 \, / \, \si{\micro\meter}$")
ax.legend()
plt.savefig("Plots/Fit_eff_mass.pdf", bbox_inches = "tight")
