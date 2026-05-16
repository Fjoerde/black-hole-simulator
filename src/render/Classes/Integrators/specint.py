import numpy as np
from numba import njit
from numba.typed import List

@njit
def init(self, specint_grid, geodesics, grav_field, obs_vel):
    if obs_vel.shape != (4,): raise ValueError("Invalid shape for obs_vel.")

    self.specint_grid = specint_grid
    self.geodesics = List(geodesics)
    self.grav_field = grav_field
    self.solve_idx = 0
    self.doppler_obs = np.empty(len(geodesics), dtype=np.float64)

    for i in range(len(geodesics)):
        x = self.geodesics[i].vals[-1,:4]
        k = self.geodesics[i].vals[-1,4:8]
        J = self.grav_field.jacobian(x); g = self.grav_field.sample_g(self.grav_field.coord_pos(x))
        g = (g @ J) @ J # Transform coordinates back to Minkowski
        self.doppler_obs[i] = (g @ k) @ obs_vel

@njit
def derivative(self, t:float, y:np.ndarray) -> np.ndarray:
    t_arr = np.array([[t]], dtype=np.float64)
    geodesic = self.geodesics[self.solve_idx]
    vals_interp = geodesic.interp(t_arr)[0]
    x = vals_interp[:4]; k = vals_interp[4:8]
    vel = vals_interp[9:13]; temp = max(vals_interp[13], 0); ext_coeff = max(vals_interp[14], 0)

    J = self.grav_field.jacobian(x); g = self.grav_field.sample_g(self.grav_field.coord_pos(x))
    g = (g @ J) @ J # Coordinate transformation to Minkowski
    D_src = (g @ k) @ vel; D_obs = self.doppler_obs[self.solve_idx]
    if np.abs(D_obs) < 1e-16 and np.abs(D_src) < 1e-16: D = 1
    else: D = D_src / D_obs

    h = 6.62607015e-34; c = 2.99792458e8; kB = 1.380649e-23
    wvls = self.specint_grid.pts.reshape(len(self.specint_grid.pts)) * 1e-9
    if D_obs == 0 or temp == 0: B = np.zeros(len(self.specint_grid.pts), dtype=np.float64)
    else: B = 2*h*c**2/wvls**5 * 1/(np.exp((h*c*D)/(wvls*kB*temp)) - 1) * 1e-9 # Planck's law

    dI = np.linalg.norm(k[1:]) * ext_coeff * (B - y)
    if np.isnan(dI).any(): raise ValueError("Encountered NaN values.")
    return dI

@njit
def term_cond(self, t:float, y:np.ndarray, h:float) -> bool:
    return False

@njit
def sample_func(self, y:np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.zeros(1, dtype=np.float64))

@njit
def max_step(self, t:float, y:np.ndarray, h:float) -> float:
    return np.inf
