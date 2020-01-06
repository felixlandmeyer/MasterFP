import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

x,B = np.genfromtxt("data/Magnetfeld.txt", skip_header = 2, unpack = True)
d_kante = 132#x[9]

d_kante -= min(x)
x -= min(x)

d_P = d_kante-1.36



fig, ax = plt.subplots(figsize = (5,3))
ax.scatter(x,B,color ="indianred")
ax.axvspan(d_P, d_kante, alpha=0.7, color= 'cornflowerblue')
ax.text(x = (d_kante+d_P)/2 , y = (min(B)+max(B))/2,s = r"Probe",weight='bold', horizontalalignment='center')
ax.set_ylabel(r"Magnetfeld / mT")
ax.set_xlabel(r"Probenspalt / mm")
plt.savefig("Plots/Magnetfeldstärke.png", dpi = 300, bbox_inches = "tight")
