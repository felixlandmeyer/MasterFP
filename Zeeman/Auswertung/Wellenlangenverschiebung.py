import numpy as np
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as std)
import uncertainties.unumpy as unp

def WV(deltas, Deltas, Deltalambda):
    return 1/2 * (deltas/Deltas) * Deltalambda

DL_rot = 4.9 * 10**(-11)
DL_blau = 2.7 * 10**(-11)

def Fehler(deltas, Deltas, Deltalambda):
    return 0.5 * Deltalambda * np.sqrt((1/Deltas * 3)**2 + ((deltas / (Deltas**2))*3)**2)

Deltas_rot = np.array([33,35,34,32,34,33,31,33,34,31])
deltas_rot = np.array([16,17,16,15,15,16,16,18,18,15])

print('Wellenlängenverschiebung rot:', WV(deltas_rot,Deltas_rot,DL_rot))
print('Fehler WV rot:', Fehler(deltas_rot,Deltas_rot, DL_rot))


def Fehlerblau(deltas, Deltas, Deltalambda):
    return 0.5 * Deltalambda * np.sqrt((1/Deltas * 5)**2 + ((deltas / Deltas**2)*5)**2)

Deltas_blau = np.array([47,48,50,47,45,46,45,43,44,42])
deltas_blau_sigma = np.array([26,26,24,25,23,26,23,25,22,23])
deltas_blau_pi = np.array([16,19,20,19,19,17,21,16,20,19])

print('Wellenlängenverschiebung blau sigma:', WV(deltas_blau_sigma,Deltas_blau,DL_blau))
print('Fehler WV blau sigma:', Fehlerblau(deltas_blau_sigma,Deltas_blau, DL_blau))
print('Wellenlängenverschiebung blau pi:', WV(deltas_blau_pi,Deltas_blau,DL_blau))
print('Fehler WV blau pi:', Fehlerblau(deltas_blau_pi,Deltas_blau, DL_blau))

WV_rot = [1.18787879e-11, 1.19000000e-11, 1.15294118e-11, 1.14843750e-11,1.08088235e-11, 1.18787879e-11, 1.26451613e-11, 1.33636364e-11, 1.29705882e-11, 1.18548387e-11]
WV_rot_err = [2.47525873e-12, 2.33460918e-12, 2.38916890e-12, 2.53669704e-12,2.36279664e-12, 2.47525873e-12, 2.66814385e-12, 2.53705884e-12 ,2.44602202e-12, 2.63394254e-12]

lambdas_rot = unp.uarray(WV_rot, WV_rot_err)
mean_lambdas_rot = lambdas_rot.sum() / 10
print('Mittelwert rot = {:.2u}'.format(mean_lambdas_rot))

WV_blausig = [7.46808511e-12, 7.31250000e-12, 6.48000000e-12, 7.18085106e-12,6.90000000e-12, 7.63043478e-12, 6.90000000e-12, 7.84883721e-12,6.75000000e-12, 7.39285714e-12]
WV_blausig_err =[1.64127353e-12, 1.59929813e-12, 1.49746586e-12, 1.62670192e-12,1.68457050e-12, 1.68556621e-12, 1.68457050e-12, 1.81579455e-12, 1.71516578e-12, 1.83234488e-12]
WV_blaupi = [4.59574468e-12, 5.34375000e-12, 5.40000000e-12, 5.45744681e-12,5.70000000e-12, 4.98913043e-12, 6.30000000e-12, 5.02325581e-12,6.13636364e-12,6.10714286e-12]
WV_blaupi_err =[1.51710807e-12, 1.51241127e-12, 1.45399450e-12, 1.54908275e-12,1.62822330e-12, 1.56439223e-12, 1.65529454e-12, 1.67491554e-12,1.68513554e-12, 1.76394345e-12]


lambdas_blausig = unp.uarray(WV_blausig, WV_blausig_err)
mean_lambdas_blausig = lambdas_blausig.sum() / 10
lambdas_blaupi = unp.uarray(WV_blaupi, WV_blaupi_err)
mean_lambdas_blaupi = lambdas_blaupi.sum() / 10
print('Mittelwert blau sigma = {:.2u}'.format(mean_lambdas_blausig))
print('Mittelwert blau pi = {:.2u}'.format(mean_lambdas_blaupi))
