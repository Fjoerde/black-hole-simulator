# A place for testing out functions/methods that would not otherwise be directly called.

import numpy as np
from numba import prange
from PIL import Image

from render.Classes.base import *
from render.Classes.tags import *
from render.Classes.classes import Shape, Hittable, ColField, GravField, RenderSettings
from render.rendering import render_seq, trace
from render.diagnostics import look_ray, check_Gamma

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

scene = [Hittable(tag=HITTABLE_LIGHT,
         shape=Shape(pos=Vec(10,0,0), rot=(np.pi/2,-np.pi/20,0), tag=SHAPE_ANNULUS, r_in=2.5, r_out=5, height=0.1),
         col_field=ColField(tag=COLFIELD_CONSTCOL, col=Vec(1,1,1)))] # Accretion disc
bg = np.array(Image.open("Images/background2.jpg")) / 255.0
settings = RenderSettings(w=1280, h=720, cam_pos=Vec(0,0,0), cam_dir=Vec(1,0,0), scene=scene, 
                          background=bg, bg_rad=20,
                          grav_field=GravField(tag=GRAVFIELD_KERRNEWMAN, pos=Vec(10,0,0), M=0.5, ax=Vec(0,0,1), J=0.15, Q=0.1))

# x = [-1.38016832e+01, 7.73420547e+00, 4.10307858e-04, 4.41848130e+00]
# print(settings.grav_field.sample_Gamma(x))



for i in prange(600):
    col = trace(settings.cam_pos, settings.ray_dir_px(638,i), settings)
    print(i, col.np_array())


# col = trace(settings.cam_pos, settings.ray_dir_px(638,6), settings)
# print(col.np_array())


