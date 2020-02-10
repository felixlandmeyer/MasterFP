import numpy as np
import pandas as pd
import uncertainties as unc
import matplotlib as mpl
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
from uncertainties import ufloat
from scipy.optimize import curve_fit

Druck,M1,M2,M3,M4 = np.genfromtxt("data/Druck.txt", skip_header = 2, unpack = True)
Druck_all = Druck
###### je nachdem kann man die Rechung vllt auch mit einem Datensatz machen, der aus einem Mittelwert aller anderen Datensätze besteht
N_mean_noms = np.array([np.nanmean(np.array([M1[i],M2[i],M3[i],M4[i]])) for i in range(len(M2))])
N_mean_stds = np.array([np.nanstd(np.array([M1[i],M2[i],M3[i],M4[i]])) for i in range(len(M2))])
N_mean = unp.uarray(N_mean_noms, N_mean_stds)

df = pd.DataFrame({"Druck":Druck,"N_mean":N_mean, "M1":M1, "M2":M2, "M3":M3, "M4":M4})

L = ufloat(100e-3,0.1e-3)
T_measured = np.array([20.8,23.3,22.4,22.2,21.9,21.7,22.7]) + 273.15
T = ufloat(np.mean(T_measured), np.std(T_measured))
print(T)
print(T-273.15)
#### für n wird die Steigung #Nulldurchgänge/Druck benötigt
#### Steigung für #Nulldurchgänge/Druck ########
def f(x,m,b):
    return m * x + b
x = np.linspace(min(Druck), max(Druck), 1000)

##### Miriams Rechungen sind näher am Literaturwert
##### Steigung M1
params_1, cov = curve_fit(f, df.dropna(subset=['Druck', 'M1'])["Druck"], df.dropna(subset=['Druck', 'M1'])["M1"])
errors_1 = np.sqrt(np.diag(cov))

##### Steigung M2
params_2, cov = curve_fit(f, df.dropna(subset=['Druck', 'M2'])["Druck"], df.dropna(subset=['Druck', 'M2'])["M2"])
errors_2 = np.sqrt(np.diag(cov))

##### Steigung M3
params_3, cov = curve_fit(f, df.dropna(subset=['Druck', 'M3'])["Druck"], df.dropna(subset=['Druck', 'M3'])["M3"])
errors_3 = np.sqrt(np.diag(cov))

##### Steigung M4
params_4, cov = curve_fit(f, df.dropna(subset=['Druck', 'M4'])["Druck"], df.dropna(subset=['Druck', 'M4'])["M4"])
errors_4 = np.sqrt(np.diag(cov))

params_list = [params_1, params_2, params_3, params_4]


########## Bestimmung wie Miriam ##########
def Miriam_n(Druck,M,L = L, Lambda = 632.99e-9):
    n = Lambda/L * M + 1
    params, cov = curve_fit(f, Druck, noms(n))
    errors = np.sqrt(np.diag(cov))
    params = unp.uarray(params,errors)
    T_0 = 15 + 273.15
    Steigung_norm = params[0] * T/T_0
    atm =  1.01325e3
    n_atm = Steigung_norm * atm + params[1]
    return [params,Druck,n,n_atm]



fit_parameter = np.array([
        Miriam_n(df.dropna(subset=['Druck', 'M1'])["Druck"], df.dropna(subset=['Druck', 'M1'])["M1"])[0],
        Miriam_n(df.dropna(subset=['Druck', 'M2'])["Druck"], df.dropna(subset=['Druck', 'M2'])["M2"])[0],
        Miriam_n(df.dropna(subset=['Druck', 'M3'])["Druck"], df.dropna(subset=['Druck', 'M3'])["M3"])[0],
        Miriam_n(df.dropna(subset=['Druck', 'M4'])["Druck"], df.dropna(subset=['Druck', 'M4'])["M4"])[0],
        ])

Drücke = np.array([
        Miriam_n(df.dropna(subset=['Druck', 'M1'])["Druck"], df.dropna(subset=['Druck', 'M1'])["M1"])[1],
        Miriam_n(df.dropna(subset=['Druck', 'M2'])["Druck"], df.dropna(subset=['Druck', 'M2'])["M2"])[1],
        Miriam_n(df.dropna(subset=['Druck', 'M3'])["Druck"], df.dropna(subset=['Druck', 'M3'])["M3"])[1],
        Miriam_n(df.dropna(subset=['Druck', 'M4'])["Druck"], df.dropna(subset=['Druck', 'M4'])["M4"])[1],
        ])
print(Drücke)

n = np.array([
        Miriam_n(df.dropna(subset=['Druck', 'M1'])["Druck"], df.dropna(subset=['Druck', 'M1'])["M1"])[2],
        Miriam_n(df.dropna(subset=['Druck', 'M2'])["Druck"], df.dropna(subset=['Druck', 'M2'])["M2"])[2],
        Miriam_n(df.dropna(subset=['Druck', 'M3'])["Druck"], df.dropna(subset=['Druck', 'M3'])["M3"])[2],
        Miriam_n(df.dropna(subset=['Druck', 'M4'])["Druck"], df.dropna(subset=['Druck', 'M4'])["M4"])[2],
        ])


n_atm_miriam = np.array([
        Miriam_n(df.dropna(subset=['Druck', 'M1'])["Druck"], df.dropna(subset=['Druck', 'M1'])["M1"])[3],
        Miriam_n(df.dropna(subset=['Druck', 'M2'])["Druck"], df.dropna(subset=['Druck', 'M2'])["M2"])[3],
        Miriam_n(df.dropna(subset=['Druck', 'M3'])["Druck"], df.dropna(subset=['Druck', 'M3'])["M3"])[3],
        Miriam_n(df.dropna(subset=['Druck', 'M4'])["Druck"], df.dropna(subset=['Druck', 'M4'])["M4"])[3],
        ])

print(fit_parameter[:,0],"\n", fit_parameter[:,1],"\n", n_atm_miriam)
print("Mean:", np.mean(n_atm_miriam))

n_lit = 1.000277

Abw = 1 - (n_lit-1)/(np.mean(n_atm_miriam)-1)

print("Abweichung von Literaturwert in Prozent:", Abw*100)

measure_list = [M1,M2,M3,M4]

table = {"Druck": Druck, "M1": M1, "n1": n[0], "M2": M2, "n2": n[1], "M3": M3, "n3": n[2], "M4": M4, "n4": n[3]}
with open('outputdata.tex', 'w') as tf:
     tf.write(pd.DataFrame(table).to_latex(index=False,na_rep=""))


labels = ["Messreihe 1", "Messreihe 2", "Messreihe 3", "Messreihe 4"]
markers = ["+","*","1","x"]
cmap = mpl.cm.get_cmap('viridis')
colors = cmap([0,0.33,0.66,1])

fig, ax = plt.subplots(figsize = (5,3))
for measure,p,fit,label,marker,color in zip(measure_list,Drücke,params_list,labels,markers,colors):
    ax.scatter(Druck,measure,s = 50,color = color,marker = marker, alpha =.8, label = label)
    ax.plot(p,f(p, *fit),color = color,alpha =.8,label = "")
#ax.errorbar(Druck, noms(N_mean), yerr=stds(N_mean), fmt = "o" , c = "k")
ax.set_xlabel(r"Druck / mbar")
ax.set_ylabel(r"# Nulldurchgänge")
ax.legend(loc = "upper left")
plt.savefig("Plots/Messwerte.png", bbox_inches = "tight", dpi = 300)

fig, ax = plt.subplots(figsize = (5,3))

for Druck,n,params,label,marker,color in zip(Drücke,n,fit_parameter,labels,markers,colors):
    ax.scatter(Druck,noms(n),s = 50,color = color,marker = marker, alpha =.8, label = label)
    ax.plot(Druck,f(Druck, *noms(params)),color = color,alpha =.8, label = "")
    ax.set_ylim(1, 1.00029)
#ax.errorbar(Druck, noms(N_mean), yerr=stds(N_mean), fmt = "o" , c = "k")
ax.legend(loc = "upper left")
ax.set_xlabel(r"Druck / mbar")
ax.set_ylabel(r"Brechungsindex")
plt.savefig("Plots/Brechungsindex.png", bbox_inches = "tight", dpi = 300)
