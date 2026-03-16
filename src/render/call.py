import numpy as np
from PIL import Image
import time
from render.Classes.base import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Set gravitational field and precompute before importing other classes
print(f"Initializing...\n")
t1 = time.perf_counter()

from render.Classes.GravField.grav_lib import Minkowski, Schwarzschild, Kerr
# Schwarzschild(X=0, Y=0, Z=0, M=0.5).use(filename="funcs.py")

from render.Classes.GravField.precomp import precomp_all
cam_pos = Vec(-10,0,0); bg_rad = 20.
# precomp_all(save=True, k=12, center=cam_pos, bg_rad=bg_rad, n=10)

# Import other classes and initialize & render
from render.Classes.tags import *
from render.Classes.classes import Shape, Hittable, ColField, RenderSettings
from render.rendering import render_seq

scene = [Hittable(tag=HITTABLE_LIGHT,
         shape=Shape(pos=Vec(0,0,0), rot=(np.pi/2,-np.pi/20,0), tag=SHAPE_ANNULUS, r_in=2.5, r_out=5, height=0.1),
         col_field=ColField(tag=COLFIELD_CONSTCOL, col=Vec(1,1,1)))] # Accretion disc
bg = np.array(Image.open("Images/background2.jpg")) / 255.0
grid = np.load("Classes/GravField/Points/grid_pts.npy")
neighbors = np.load("Classes/GravField/Points/neigh_pts.npy").astype(int)
g_grid = np.load("Classes/GravField/Points/metric_grid.npy")
Gamma_grid = np.load("Classes/GravField/Points/chr_sym_grid.npy")
settings = RenderSettings(w=800, h=600, cam_pos=cam_pos, cam_dir=Vec(1,0,0), scene=scene, background=bg, bg_rad=bg_rad,
                          grid=grid, neighbors=neighbors, g_grid=g_grid, Gamma_grid=Gamma_grid)

t2 = time.perf_counter()
print(f"Initialization finished in {t2-t1:.4f} s\n")

print("Rendering...")
img = render_seq(settings)
filename = "test15.png"
img.save(f"Images/{filename}") 
    
t3 = time.perf_counter()
print(f"Rendering finished in {(t3-t2)/60:.4f} min\n")
print(f"Saved '{filename}'")
