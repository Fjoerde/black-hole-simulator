# A place for testing out functions/methods that would not otherwise be directly called.
from PIL import Image
from render.Classes.base import *
from render.Classes.tags import *
from render.Classes.classes import Shape, Hittable, ColField, RenderSettings
from render.rendering import *
from numba import prange
import render.Classes.GravField.methods as GravMethod
from render.Classes.GravField.precomp import *
from scipy.spatial import cKDTree

scene = [Hittable(tag=HITTABLE_LIGHT,
         shape=Shape(pos=Vec(0,0,0), rot=(np.pi/2,-np.pi/20,0), tag=SHAPE_ANNULUS, r_in=2.5, r_out=5, height=0.1),
         col_field=ColField(tag=COLFIELD_CONSTCOL, col=Vec(1,1,1)))] # Accretion disc
bg = np.array(Image.open("Images/background2.jpg")) / 255.0
settings = RenderSettings(w=800, h=600, cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), scene=scene, background=bg, bg_rad=20, threshold=1e-6) # If no object, simply omit scene argument
settings.precomp_all(n=30)