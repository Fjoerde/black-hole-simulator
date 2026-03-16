# A place for testing out functions/methods that would not otherwise be directly called.
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from render.Classes.base import *
from render.Classes.GravField.grav_lib import Schwarzschild
Schwarzschild(X=0, Y=0, Z=0, M=0.5).use(filename="funcs.py")
import render.Classes.GravField.precomp as GravComp
cam_pos = Vec(-10,0,0); bg_rad = 20
GravComp.precomp_all(save=True, k=20, center=cam_pos, bg_rad=bg_rad)
