import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

x,B = np.genfromtxt("data/Magnetfeld.txt", skip_header = 2, unpack = True)
