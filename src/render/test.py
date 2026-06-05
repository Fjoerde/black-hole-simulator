# A place for testing out functions/methods that would not otherwise be directly called.

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# from Classes.math import *
# from Classes.int_and_settings import *
# from diagnostics import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

spec_vals = np.load("Data/doppler_vals.npy")
labels = [r"$\beta=-0.5$", r"$\beta=-0.25$", r"$\beta=0$", r"$\beta=0.25$", r"$\beta=0.5$"]
params = { # For easier viewing of the graph
    "axes.labelsize": 23,
    "font.size": 23,
    "legend.fontsize": 18,
    "xtick.labelsize": 23,
    "ytick.labelsize": 23,
    "figure.figsize": [12, 8]
}
plt.rcParams.update(params)
for i in range(5):
    xx = np.linspace(0, 1000, 1000)
    plt.plot(xx, spec_vals[i], label=labels[i], lw=4)
plt.xlabel(r"$\lambda~/~\mathrm{nm}$"); plt.ylabel(r"$I_{\lambda}~/~\mathrm{W~m^{-2}~sr^{-1}~nm^{-1}}$")
plt.legend()
plt.show()

"""
bg = np.array(Image.open("Images/background1.jpg")).astype(np.float64) / 255.
settings = RenderSettings(w=800, h=600, cam_pos=Vec(0,0,0), cam_vel=Vec(0,0,0), rot=(0, np.pi/2, 0), background=bg,
                          col_converter=ColConverter(Grid(Patch([np.linspace(0, 1000, 101, dtype=np.float64)]))))
geodesic = look_ray(Vec(0,0,0), Vec(1,0,0), 0, settings)
x0 = geodesic.vals[0,:4]; pos = Vec(x0[1], x0[2], x0[3])
k0 = geodesic.vals[0,4:]; k1 = geodesic.vals[-1,4:]
theta = np.acos(pos.z / pos.length()); phi = np.atan2(pos.y, pos.x)
green = def_cc.get_spec_int(np.array([0,1,0], dtype=np.float64))
spec_ints = []
spec_vals = np.empty((5, 1000))

for i in range(-2, 3):
    settings.cam_vel = Vec(0.25*i,0,0)
    spec_int = settings.doppler_spec(green, x0, k0, k1)
    col = def_cc.get_rgb(spec_int); display_col(col)
    print(f"beta = {0.25*i}, col = {(col*255).astype(np.int64)}")
    spec_ints.append(spec_int)

    xx = np.linspace(np.min(spec_int.grid.pts), np.max(spec_int.grid.pts), 1000)
    spec_vals[i+2] = spec_int.interp(xx.reshape(xx.shape[0], 1))[:,0]

np.save("Data/doppler_vals.npy", spec_vals)
"""