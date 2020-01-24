import numpy as np
import pandas as pd
import uncertainties as unc
import matplotlib as mpl
import matplotlib.pyplot as plt
from uncertainties import unumpy
from uncertainties import ufloat
from scipy.optimize import curve_fit

M = np.array([42,38,35,35,38,35,36,37,37,36])
Lambda = 632.99e-9
Theta = np.radians(11)
Theta_0 = np.radians(10)
T = 1e-3

n = (1 - Lambda * M / (2 * T * (Theta_0 * Theta)))**-1
n_mean = ufloat(np.mean(n), np.std(n)/ np.sqrt(len(n)))
np.savetxt("data/Brechungsindex_Glas.txt", np.array([M, n]).T, header = "#Nulldurchgänge Brechungsindex")
n_lit = 1.5
Abw = 1 - n_mean/n_lit
print("Abw", Abw)
print(n_mean)
