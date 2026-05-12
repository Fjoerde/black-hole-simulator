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

test1 = [Hittable(tag=HITTABLE_BLACKBODY,
                  shape=Shape(pos=Vec(0,0,0), rot=(0,0,0), tag=SHAPE_SPHERE, radius=3), temp=6500.)]
test2 = [Hittable(tag=HITTABLE_CHECKERBOARD,
                  shape=Shape(pos=Vec(0,0,0), rot=(0,0,0), tag=SHAPE_SPHERE, radius=3),
                  col1=np.array([0,0,1]), col2=np.array([0,1,0]))]
disk = [Hittable(tag=HITTABLE_BLACKBODY,
                 shape=Shape(pos=Vec(0,0,0), rot=(np.pi,-np.pi/20,0), tag=SHAPE_ANNULUS, r_in=2.5, r_out=5, height=0.25), temp=3e3)]

settings = RenderSettings(w=1280, h=720,
                          cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), cam_vel=Vec(0,0,0),
                          scene=test2)
geodesic = look_ray(Vec(-10,0,0), settings.ray_dir_px(907, 296), settings)
print(geodesic.vals[:,:4].flatten)
spec_int, col = ray_col(geodesic, Vec(0,0,0), settings)
display_specint(spec_int)
display_col(col)

