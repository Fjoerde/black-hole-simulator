# A place for testing out functions/methods that would not otherwise be directly called.

import numpy as np
from PIL import Image
from Classes.math import *
from Classes.int_and_settings import RenderSettings
from render.img_rendering import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

bg = np.array(Image.open("Images/background1.jpg")).astype(np.float64) / 255.
settings = RenderSettings(w=800, h=600, cam_pos=Vec(-5,0,0), rot=(0,np.pi/2,np.pi/4), background=bg)
rendered_img, ang_dev_img, avg_specint_img = render_img(settings)
