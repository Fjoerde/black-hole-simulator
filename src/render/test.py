# A place for testing out functions/methods that would not otherwise be directly called.
import numpy as np
from PIL import Image
from itertools import product

from Classes.math import *
from Classes.physics import *
from Classes.int_and_settings import *
from Classes.tags import *
from diagnostics import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

print("hi")
"""
def get_gas(pts):
    x, y, z = pts[:,1], pts[:,2], pts[:,3]
    r = np.sqrt(x**2 + y**2 + z**2)
    temp = 5e4 * np.exp(-r**2/10)
    ext_coeff = 1 * np.exp(-r**2/10)
    gas_params = np.zeros((len(pts), 6), dtype=np.float64); gas_params[:,0] += 1
    gas_params[:,4] = temp; gas_params[:,5] = ext_coeff
    return gas_params
grid = Grid(Patch([np.array([0], dtype=np.float64),
                   np.linspace(-30, 10, 11, dtype=np.float64),
                   np.linspace(-30, 10, 11, dtype=np.float64),
                   np.linspace(-30, 10, 11, dtype=np.float64)]))
print("hi2")
grid = grid.add_patch(Patch([np.array([0], dtype=np.float64),
                             np.linspace(-6, 6, 13, dtype=np.float64),
                             np.linspace(-6, 6, 13, dtype=np.float64),
                             np.linspace(-6, 6, 13, dtype=np.float64)]))
gas_vals = get_gas(grid.pts)
gas = Function(grid, gas_vals)
print("hi3")
print(gas.interp(np.array([[0.,-10.,0.,0.]])))
"""

# ss = GravField(tag=GRAVFIELD_SCHWARZSCHILD, pos=Vec(0,0,0), M=0.5)

bg = np.array(Image.open("Images/background1.jpg")).astype(np.float64) / 255.
settings = RenderSettings(w=800, h=600, cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), cam_vel=Vec(0.1,0,0), background=bg)
geodesic = look_ray(Vec(-10,0,0), Vec(1,0,0), settings)
gas_vals = display_gas_vals(geodesic, settings.gas, 100)
spec_int, col = ray_col(geodesic, gas_vals, Vec(0,0,0), settings)
display_col(col)
print(col)
display_specint(spec_int)
