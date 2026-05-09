import numpy as np
from numba import njit
from numba.typed import List

from render.Classes.math import Vec

@njit
def init(self, grav_field, scene, cam_pos, bg_rad, gas):
    self.grav_field = grav_field
    self.scene = List(scene)
    self.cam_pos = cam_pos
    self.bg_rad = bg_rad
    self.gas = gas

@njit
def derivative(self, _, y:np.ndarray) -> np.ndarray:
    y = np.ascontiguousarray(y)
    x, v = y[:4], y[4:8]
    chr_syms = self.grav_field.sample_Gamma(x)
    A = np.zeros(4, dtype=np.float64)
    for c in range(4): A[c] -= (chr_syms[c] @ v) @ v # Geodesic equation
    # tau = np.linalg.norm(v[1:]) * self.gas.interp(x.reshape(1, 4))[0,-1] # Optical depth
    y_new = np.concatenate((v, A))
    return np.ascontiguousarray(y_new)

@njit
def term_cond(self, _, y:np.ndarray, h:float) -> bool:
    # If reached maximum optical depth (tau >= 3 means intensity drops to <5%)
    if y[-1] >= 3: return True
    # Check if approaching a black hole
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
def sample_func(self, _, y:np.ndarray) -> np.ndarray:
    x = y[:4]; x_arr = x.reshape(1, 4)
    gas_params = self.gas.interp(x_arr)[0]
    return gas_params

@njit
def max_step(self, t:float, y:np.ndarray, _) -> float:
    speed = np.linalg.norm(self.derivative(t, y))
    dists = np.empty(len(self.scene), dtype=np.float64)
    for i in range(len(self.scene)):
        obj = self.scene[i]
        dists[i] = obj.shape.in_shape_int(self.grav_field.mink_pos(y[:4]))
    return 1.25 * min(dists) / speed
