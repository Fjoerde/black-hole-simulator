import numpy as np
from numba import njit

@njit
def init(self, specint_grid, geodesic, gas, grav_field, obs_vel):
    if obs_vel.shape != (4,): raise ValueError("Invalid shape for obs_vel.")

    self.specint_grid = specint_grid
    self.geodesic = geodesic
    self.gas = gas
    self.grav_field = grav_field

    x = self.geodesic[-1][:4]
    J = self.grav_field.jacobian(x); g = self.grav_field.sample_g(self.grav_field.coord_pos(x))
    g = (g @ J) @ J # Transform coordinates back to Minkowski
    self.doppler_obs = (g @ self.geodesic[-1][4:]) @ obs_vel

@njit
def derivative(self, t:float, y:np.ndarray) -> np.ndarray:
    X = self.geodesic.interp(t)[0]
    x = X[:4]; k = X[4:]
    vel = self.gas.vels.interp(t)[0]
    temp = self.gas.temps.interp(t)[0,0]
    ext_coeff = self.gas.ext_coeffs.interp(t)[0,0]
    
    J = self.grav_field.jacobian(x); g = self.grav_field.sample_g(self.grav_field.coord_pos(x))
    g = (g @ J) @ J # Coordinate transformation to Minkowski
    D = ((g @ k) @ vel) / self.doppler_obs # Doppler factor
    h = 6.63e-34; c = 3e8; kB = 1.38e-23
    wvls = self.grid.pts.reshape(len(self.grid.pts))
    B = 2*h*c**2/wvls**5 * 1/(np.exp((h*c*D)/(wvls*kB*temp)) - 1) # Planck's law
    return np.linalg.norm(k[1:]) * ext_coeff * (B - y)

@njit
def term_cond(self, t:float, y:np.ndarray, h:float) -> bool:
    return False

@njit
def max_step(self, t:float, y:np.ndarray, h:float) -> float:
    return np.inf
