import numpy as np
from PIL import Image
import time
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Set gravitational field before importing other classes
print("Initializing...")
t1 = time.perf_counter()

from render.Classes.GravField.grav_lib import Minkowski, Schwarzschild
# Minkowski().use()
Schwarzschild(X=0., Y=0., Z=0., M=0.5).use()

# Import other classes and initialize & render
from render.Classes.base import *
from render.Classes.tags import *
from render.Classes.classes import Shape, Hittable, ColField, RenderSettings
from render.rendering import render_seq

scene = [Hittable(tag=HITTABLE_LIGHT,
         shape=Shape(pos=Vec(0,0,0), rot=(np.pi/2,-np.pi/10,0), tag=SHAPE_ANNULUS, r_in=1.5, r_out=5., height=0.15),
         col_field=ColField(tag=COLFIELD_CONSTCOL, col=Vec(1,1,1)))] # Accretion disc
bg = np.array(Image.open("Images/background2.jpg")) / 255.0
settings = RenderSettings(w=800, h=600, cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), scene=scene, background=bg, bg_rad=20, threshold=1e-6) # If no object, simply omit scene argument

t2 = time.perf_counter()
print(f"Initialization finished in {t2-t1:.4f} s\n")

print("Rendering...")
img = render_seq(settings)
filename = "test13.png"
img.save(f"Images/{filename}") 
    
t3 = time.perf_counter()
print(f"Rendering finished in {(t3-t2)/60:.4f} min\n")
print(f"Saved '{filename}'")
