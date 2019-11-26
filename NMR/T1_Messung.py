import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

Messwerte = np.array([

0.001,1.58,
0.01,1.55,
0.03,1.53,
0.04,1.49,
0.06,1.48,
0.1,1.39,
0.11,1.36,
0.13,1.34,
0.2,1.21,
0.25,1.106,
0.3,1,
0.35,0.943,
0.4,0.837,
0.45,0.762,
0.5,0.706,
0.6,0.523,
0.7,0.425,
0.8,0.325,
0.9,0.206,
1,0.125,
1.04,-0.128,
2,-0.912,
5,-1.33,
9,-1.33
####### weitere Messpunkte liegen im Rauschen
])
tau = Messwerte[::2]
A = Messwerte[1::2]

def M(tau,M_0,T_1,M_1):
    return M_0 + np.exp(-tau/T_1) + M_1



fig,ax = plt.subplots()
ax.plot(tau,A)
ax.set_xscale("log")
#ax.set_yscale("log")
plt.show()
