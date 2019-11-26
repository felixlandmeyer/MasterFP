import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

tau,A = np.genfromtxt("Diffusion.txt", skip_header=2,unpack = True)

fig,ax = plt.subplots()
ax.scatter(tau,A)
#ax.set_xscale("log")
#ax.set_yscale("log")
plt.show()
