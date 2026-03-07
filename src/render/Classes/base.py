import numpy as np
from numba import float64
from numba.experimental import jitclass
from numba.extending import overload
import operator
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

spec_vec= [("x", float64), ("y", float64), ("z", float64)]
@jitclass(spec_vec)
class Vec: # Class of vectors
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __add__(self, vec): return Vec(self.x+vec.x, self.y+vec.y, self.z+vec.z)
    def __sub__(self, vec): return Vec(self.x-vec.x, self.y-vec.y, self.z-vec.z)
    def __mul__(self, scal): return Vec(self.x*scal, self.y*scal, self.z*scal)
    def __rmul__(self, scal): return self * scal
    def __truediv__(self, scal): return Vec(self.x/scal, self.y/scal, self.z/scal)
    def __neg__(self): return Vec(-self.x, -self.y, -self.z)
    def __eq__(self, vec): return (self.x == vec.x and self.y == vec.y and self.z == vec.z)
    def __ne__(self, vec): return (self.x != vec.x or self.y != vec.y or self.z != vec.z)

    def length(self): return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    def normal(self):
        len_ = self.length()
        if len_ < 1e-8: return Vec(0,0,0) # Avoid division by zero
        return self / len_
    def dot(self, vec): return (self.x*vec.x + self.y*vec.y + self.z*vec.z)
    def cross(self, vec): return Vec(self.y*vec.z-vec.y*self.z, self.z*vec.x-self.x*vec.z, self.x*vec.y-self.y*vec.x)
    def np_array(self): return np.array([self.x, self.y, self.z]) # Turn Vec into numpy array

    def is_normal(self):
        if (self.length() - 1.0) > 1e-6: raise ValueError("Expected normalized vector.")

@overload(operator.mul)
def vec_mul(v1, v2):
    if v1 == Vec.class_type.instance_type and isinstance(v2, Vec.class_type.instance_type): raise ValueError("Encountered multiplication of vectors.")
    elif v1 == float64 and v2 == Vec.class_type.instance_type: return lambda v1,v2: Vec(v2.x*v1, v2.y*v1, v2.z*v1)
    elif v1 == Vec.class_type.instance_type and v2 == float64: return lambda v1,v2: Vec(v1.x*v2, v1.y*v2, v1.z*v2)


spec_bb = [("min_corner", Vec.class_type.instance_type), ("max_corner", Vec.class_type.instance_type),
           ("pt", Vec.class_type.instance_type), ("pos", Vec.class_type.instance_type), ("dir", Vec.class_type.instance_type)]
@jitclass(spec_bb)
class BoundingBox:
    def __init__(self, min_corner:Vec, max_corner:Vec):
        self.min_corner = min_corner
        self.max_corner = max_corner
        if not (self.max_corner.x >= self.min_corner.x and self.max_corner.y >= self.min_corner.y and self.max_corner.z >= self.min_corner.z):
            raise ValueError("Corners are not the min/max corners of the bounding box.")
        
    def in_bb(self, pt:Vec) -> bool:
        """Detects if a point is within the bounding box."""
        
        min, max = self.min_corner, self.max_corner
        return (min.x < pt.x < max.x) and (min.y < pt.y < max.y) and (min.z < pt.z < max.z)
    
    def hit_bb(self, pos:Vec, dir:Vec) -> bool:
        """Detects if a straight line starting oriented in the direction of dir will hit the bounding box."""

        dir.is_normal()
        min_np, max_np, pos_np, dir_np = self.min_corner.np_array(), self.max_corner.np_array(), pos.np_array(), dir.np_array()
        inv_dir = np.where(np.abs(dir_np) > 1e-6, 1./dir_np, np.sign(dir_np)*1e30)
        t0, t1 = (min_np-pos_np)*inv_dir, (max_np-pos_np)*inv_dir
        t_enter, t_exit = np.max(np.minimum(t0, t1)), np.min(np.maximum(t0, t1))
        return (t_enter <= t_exit and t_exit >= 0)
    