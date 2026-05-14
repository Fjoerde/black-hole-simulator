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

spec_int = bb_specint(4000)
col = def_cc.get_rgb(spec_int); print(col)
display_col(col)
display_specint(spec_int)