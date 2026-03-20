import numpy as np
from numba import njit

@njit
def init(self, pos, M):
    self.pos = pos
    self.M = M

@njit
def sample_g(self, x):
    t, r, theta, phi = x
    g = np.diag(np.array([-(1-2*self.M/r), 1/(1-2*self.M/r), r**2, r**2*np.sin(theta)**2], dtype=np.float64))
    return g

@njit
def sample_Gamma(self, x):
    t, r, theta, phi = x
    Gamma = np.zeros((4,4,4), dtype=np.float64)
    Gamma[0,0,1] = Gamma[0,1,0] = self.M/r**2 * (1-2*self.M/r)**(-1)
    Gamma[1] = np.diag(np.array([self.M/r**2*(1-2*self.M/r), -self.M/r**2*(1-2*self.M/r)**(-1), -r*(1-2*self.M/r), -r*np.sin(theta)**2*(1-2*self.M/r)]))
    Gamma[2,1,2] = Gamma[2,2,1] = Gamma[3,1,3] = Gamma[3,3,1] = 1/r
    Gamma[2,3,3] = -np.sin(theta)*np.cos(theta)
    Gamma[3,2,3] = Gamma[3,3,2] = 1/np.tan(theta)
    return Gamma

@njit
def coord_pos(self, x):
    T, X, Y, Z = x

    out = np.zeros(4, dtype=np.float64)
    out[0] = T
    out[1] = r = np.sqrt((X-self.pos.x)**2 + (Y-self.pos.y)**2 + (Z-self.pos.z)**2)
    out[2] = np.acos((Z-self.pos.z) / r)
    out[3] = np.atan2(Y-self.pos.y, X-self.pos.x)
    return out

@njit
def mink_pos(self, x):
    T, R, THETA, PHI = x

    out = np.zeros(4, dtype=np.float64)
    out[0] = T
    out[1] = R * np.sin(THETA) * np.cos(PHI)
    out[2] = R * np.sin(THETA) * np.sin(PHI)
    out[3] = R * np.cos(THETA)
    return out

@njit
def jacobian(self, x):
    T, X, Y, Z = x
    J = np.zeros((4,4), dtype=np.float64)
    J[0,0] = 1
    J[1] = np.array([0, X-self.pos.x, Y-self.pos.y, Z-self.pos.z]) / np.sqrt((X-self.pos.x)**2 + (Y-self.pos.y)**2 + (Z-self.pos.z)**2)
    J[2,1:3] = np.array([X-self.pos.x, Y-self.pos.y]) * (Z-self.pos.z)/(np.sqrt((X-self.pos.x)**2 + (Y-self.pos.y)**2)*((X-self.pos.x)**2 + (Y-self.pos.y)**2 + (Z-self.pos.z)**2))
    J[2,3] = -np.sqrt((X-self.pos.x)**2 + (Y-self.pos.y)**2)/((X-self.pos.x)**2 + (Y-self.pos.y)**2 + (Z-self.pos.z)**2)
    J[3] = np.array([0, -(Y-self.pos.y), X-self.pos.x, 0]) / ((X-self.pos.x)**2 + (Y-self.pos.y)**2)
    return J

@njit
def jacobian_inv(self, x):
    T, R, THETA, PHI = x
    J = np.zeros((4,4), dtype=np.float64)
    J[0,0] = 1
    J[1] = np.array([0, np.sin(THETA)*np.cos(PHI), R*np.cos(THETA)*np.cos(PHI), -R*np.sin(THETA)*np.sin(PHI)])
    J[2] = np.array([0, np.sin(THETA)*np.sin(PHI), R*np.cos(THETA)*np.sin(PHI), R*np.sin(THETA)*np.cos(PHI)])
    J[3] = np.array([0, np.cos(THETA), -R*np.sin(THETA), 0])
    return J