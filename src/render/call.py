import numpy as np
from PIL import Image
import time
from render.Classes.base import *

from render.Classes.tags import *
from render.Classes.classes import Shape, Hittable, ColField, GravField, RenderSettings
from render.rendering import render_seq

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

print("Initializing...")
t1 = time.perf_counter()
scene = [Hittable(tag=HITTABLE_LIGHT,
         shape=Shape(pos=Vec(10,0,0), rot=(np.pi/2,-np.pi/20,0), tag=SHAPE_ANNULUS, r_in=2.5, r_out=5, height=0.1),
         col_field=ColField(tag=COLFIELD_CONSTCOL, col=Vec(1,1,1)))] # Accretion disc
bg = np.array(Image.open("Images/background2.jpg")) / 255.0
settings = RenderSettings(w=1280, h=720, cam_pos=Vec(0,0,0), cam_dir=Vec(1,0,0), scene=scene, 
                          background=bg, bg_rad=20,
                          grav_field=GravField(tag=GRAVFIELD_KERRNEWMAN, pos=Vec(10,0,0), M=0.5, ax=Vec(0,0,1), J=0., Q=0.25))

t2 = time.perf_counter()
print(f"Initialization finished in {t2-t1:.4f} s\n")

print("Rendering...")
img = render_seq(settings)
filename = "test18.png"
img.save(f"Images/{filename}") 
    
t3 = time.perf_counter()
print(f"Rendering finished in {(t3-t2)/60:.4f} min\n")
print(f"Saved '{filename}'")
