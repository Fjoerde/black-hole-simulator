import numpy as np
from numba import njit
from numba.typed import List

@njit
def init(self, specint_grid, geodesic, gas_val, grav_field, obs_vel):
    if obs_vel.shape != (4,): raise ValueError("Invalid shape for obs_vel.")

    self.specint_grid = specint_grid
    self.geodesic = geodesic
    self.gas_val = gas_val
    self.grav_field = grav_field

    x = self.geodesic.vals[-1,:4]
    k = self.geodesic.vals[-1,4:8]
    J = self.grav_field.jacobian(x); g = self.grav_field.sample_g(self.grav_field.coord_pos(x))
    g = J.T @ g @ J # Transform coordinates back to Minkowski
    self.doppler_obs = (g @ k) @ obs_vel

@njit
def derivative(self, t:float, y:np.ndarray) -> np.ndarray:
    t_arr = np.array([[t]], dtype=np.float64)

    geodesic_val = self.geodesic.interp(t_arr)[0]
    x = geodesic_val[:4]; k = geodesic_val[4:8]
    gas_val = self.gas_val.interp(t_arr)[0]
    vel = gas_val[:4]; temp = max(gas_val[4], 0); ext_coeff = max(gas_val[5], 0)

    J = self.grav_field.jacobian(x); g = self.grav_field.sample_g(self.grav_field.coord_pos(x))
    g = J.T @ g @ J # Coordinate transformation to Minkowski
    D_src = (g @ k) @ vel
    if max(np.abs(self.doppler_obs), np.abs(D_src)) < 1e-16: D = 1
    else: D = D_src / self.doppler_obs

    h = 6.62607015e-34; c = 2.99792458e8; kB = 1.380649e-23
    wvls = self.specint_grid.pts.reshape(len(self.specint_grid.pts)) * 1e-9
    if self.doppler_obs == 0 or temp == 0: B = np.zeros(len(self.specint_grid.pts), dtype=np.float64)
    else: B = 2*h*c**2/wvls**5 * 1/(np.exp((h*c*D)/(wvls*kB*temp)) - 1) * 1e-9 # Planck's law

    dI = np.linalg.norm(k[1:]) * ext_coeff * (B - y)
    if np.isnan(dI).any(): raise ValueError("Encountered NaN values.")
    return dI

@njit
def term_cond(self, t:float, y:np.ndarray, h:float) -> bool:
    return False

@njit
def max_step(self, t:float, y:np.ndarray, h:float) -> float:
    return np.inf
