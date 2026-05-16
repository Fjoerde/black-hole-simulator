# A place for testing out functions/methods that would not otherwise be directly called.
import numpy as np
from PIL import Image

from Classes.math import *
from Classes.physics import *
from Classes.int_and_settings import *
from Classes.tags import *
from diagnostics import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def get_gas(pts):
    x, y, z = pts[:,1], pts[:,2], pts[:,3]
    r = np.sqrt(x**2 + y**2 + z**2)
    temp = 1e5 * np.exp(-r**2/20)
    ext_coeff = 1 * np.exp(-r**2/20)
    gas_params = np.zeros((len(pts), 6), dtype=np.float64); gas_params[:,0] += 1
    gas_params[:,4] = temp; gas_params[:,5] = ext_coeff
    return gas_params
grid = Grid(Patch([np.array([0], dtype=np.float64),
                   np.linspace(-5, 5, 11, dtype=np.float64),
                   np.linspace(-5, 5, 11, dtype=np.float64),
                   np.linspace(-5, 5, 11, dtype=np.float64)]))
grid = grid.add_patch(Patch([np.array([0], dtype=np.float64),
                             np.linspace(-3, 3, 13, dtype=np.float64),
                             np.linspace(-3, 3, 13, dtype=np.float64),
                             np.linspace(-3, 3, 13, dtype=np.float64)]))
gas_vals = get_gas(grid.pts)
gas = Function(grid, gas_vals)

ss = GravField(tag=GRAVFIELD_SCHWARZSCHILD, pos=Vec(0,0,0), M=0.5)
settings = RenderSettings(w=800, h=600, cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), cam_vel=Vec(0,0,0),
                          grav_field=ss)
geodesic = look_ray(Vec(-10,0,5), Vec(1,0,0), settings)
print(geodesic.vals[0,8])

"""
grid_geodesic = geodesic.grid; n = len(grid_geodesic.pts)
print(geodesic.vals[:,1:4])
temp_vals = geodesic.vals[:,13].reshape(n, 1); ext_coeff_vals = geodesic.vals[:,14].reshape(n, 1)
temp_geodesic = Function(grid_geodesic, np.ascontiguousarray(temp_vals))
ext_coeff_geodesic = Function(grid_geodesic, np.ascontiguousarray(ext_coeff_vals))
g_max = np.max(np.ascontiguousarray(grid_geodesic.pts))
plot_func(temp_geodesic, np.array([0], dtype=np.float64), np.array([g_max], dtype=np.float64))
plot_func(ext_coeff_geodesic, np.array([0], dtype=np.float64), np.array([g_max], dtype=np.float64))

spec_int, col = ray_col(geodesic, Vec(0,0,0), settings)
print(col)
display_col(col); display_specint(spec_int)
"""



# Geodesic integration takes too long, investigate
# Spectral intensity integration seems to just not commence, investigate
# Final image produced is way too granular, investigate