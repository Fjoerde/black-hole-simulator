# A place for testing out functions/methods that would not otherwise be directly called.

import numpy as np
from PIL import Image
from Classes.math import *
from Classes.physics import GravField
from Classes.int_and_settings import RenderSettings
from Classes.tags import *
from diagnostics import *
from rendering import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))


bg = np.array(Image.open("Images/background1.jpg")).astype(np.float64) / 255.
ss = GravField(tag=GRAVFIELD_SCHWARZSCHILD, pos=Vec(0,0,0), M=0.5)
settings = RenderSettings(w=1280, h=720, cam_pos=Vec(-10,0,5), cam_dir=Vec(10,0,-5).normal(), cam_vel=Vec(0,0,0), background=bg,grav_field=ss)
geodesic = look_ray(settings.cam_pos, settings.ray_dir_px(50, 20), settings)
