import numpy as np
from numba import njit
from numba.typed import List

from render.Classes.base import Vec

@njit
def init(self, grav_field, scene, cam_pos, gas):
    self.grav_field = grav_field
    self.scene = List(scene)
    self.cam_pos = cam_pos
    self.gas = gas

@njit
def derivative(self, _, y:np.ndarray) -> np.ndarray:
    x, v = y[:4], y[4:8]
    chr_syms = self.sample_Gamma(x)
    A = np.zeros(4)
    """
    for c in range(4):
        for a in range(4):
            for b in range(4):
                A[c] -= chr_syms[c,a,b] * v[a] * v[b]
    """
    for c in range(4): A[c] -= (chr_syms[c] @ v) @ v
    tau = np.linalg.norm(v[1:]) * self.gas.ext_coeff.interp(x) # Optical depth
    return np.concatenate((v, A, [tau]))

@njit
def term_cond(self, _, y:np.ndarray, h:float) -> bool:
    # If reached maximum optical depth (tau >= 3 means intensity drops to <5%)
    if y[-1] >= 3: return True
    # Check if it is approaching a black hole
    if h < 1e-4:
        if np.max(self.grav_field.sample_g(y[:4])) > 200: return True
    # Check if escaped the background box
    pos_mink = self.grav_field.mink_pos(y[:4]) 
    pt = Vec(pos_mink[1], pos_mink[2], pos_mink[3])
    if (pt - self.cam_pos).length() > self.bg_rad: return True
    # If the distance of closest approach is < 1e-4 or already inside an object
    for i in range(len(self.scene)):
        obj = self.scene[i]
        d = obj.shape.in_shape_int(self.grav_field.mink_pos(y[:4]))
        if np.abs(d) < 1e-4: return True
    return False

@njit
def max_step(self, t:float, y:np.ndarray, _) -> float:
    speed = np.linalg.norm(self.derivative(t, y))
    dists = np.empty(len(self.scene), dtype=np.float64)
    for i in range(len(self.scene)):
        obj = self.scene[i]
        dists[i] = obj.shape.in_shape_int(self.grav_field.mink_pos(y[:4]))
    return 1.25 * min(dists) / speed
