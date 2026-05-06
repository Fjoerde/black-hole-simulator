import numpy as np
from numba import njit

@njit
def init(self, pos, M, ax, J):
    self.pos = pos
    self.M = M
    self.J = J

    theta, phi = np.acos(ax.z), np.atan2(ax.y, ax.x)
    rot = np.zeros((4,4), dtype=np.float64)
    rot[0,0] = 1
    rot[1,1] = np.cos(theta)*np.cos(phi)
    rot[1,2] = np.cos(theta)*np.sin(phi)
    rot[1,3] = -np.sin(theta)
    rot[2,1] = -np.sin(phi)
    rot[2,2] = np.cos(phi)
    rot[3,1] = np.sin(theta)*np.cos(phi)
    rot[3,2] = np.sin(theta)*np.sin(phi)
    rot[3,3] = np.cos(theta)

    self.rot = np.ascontiguousarray(rot)
    self.rot_inv = np.ascontiguousarray(np.linalg.inv(self.rot))

@njit
def sample_g(self, x):
    t, r, theta, phi = x
    a = self.J / self.M
    rho = r**2 + a**2*np.cos(theta)**2
    delta = r**2 - 2*self.M*r + a**2
    g = np.zeros((4,4), dtype=np.float64)
    g[0,0] = -(1-2*self.M*r/rho)
    g[0,3] = g[3,0] = -2*self.M*a*r*np.sin(theta)**2/rho
    g[1,1] = rho/delta
    g[2,2] = rho
    g[3,3] = (r**2+a**2+2*self.M*a**2*r*np.sin(theta)**2/rho)*np.sin(theta)**2
    return np.ascontiguousarray(g)

@njit
def sample_Gamma(self, x):
    t, r, theta, phi = x
    a = self.J / self.M
    sigma = r**2 + a**2*np.cos(theta)**2
    delta = r**2 - 2*self.M*r + a**2

    Gamma = np.zeros((4,4,4), dtype=np.float64)
    Gamma[0,0,1] = Gamma[0,1,0] = 2*self.M*(r**2+a**2)*(r**2-a**2*np.cos(theta)**2)/(2*sigma**2*delta)
    Gamma[0,0,2] = Gamma[0,2,0] = -2*self.M*a**2*r*np.sin(theta)*np.cos(theta)/sigma**2
    Gamma[0,1,3] = Gamma[0,3,1] = self.M*a*np.sin(theta)**2*(a**2*np.cos(theta)**2*(a**2-r**2)-r**2*(a**2+3*r**2))/(sigma**2*delta)
    Gamma[0,2,3] = Gamma[0,3,2] = 2*self.M*a**3*r*np.sin(theta)**3*np.cos(theta)/sigma**2
    Gamma[1,0,0] = self.M*delta*(r**2-a**2*np.cos(theta)**2)/sigma**3
    Gamma[1,0,3] = Gamma[1,3,0] = -self.M*delta*a*np.sin(theta)**2*(r**2-a**2*np.cos(theta)**2)/sigma**3
    Gamma[1,1,1] = (r*a**2*np.sin(theta)**2-self.M*(r**2-a**2*np.cos(theta)**2))/(sigma*delta)
    Gamma[1,1,2] = Gamma[1,2,1] = -a**2*np.sin(theta)*np.cos(theta)/sigma
    Gamma[1,2,2] = -r*delta/sigma
    Gamma[1,3,3] = delta*np.sin(theta)**2*(self.M*a**2*np.sin(theta)**2*(r**2-a**2*np.cos(theta)**2)-r*sigma**2)/sigma**3
    Gamma[2,0,0] = -2*self.M*a**2*r*np.sin(theta)*np.cos(theta)/sigma**3
    Gamma[2,0,3] = Gamma[2,3,0] = 2*self.M*a*r*(r**2+a**2)*np.sin(theta)*np.cos(theta)/sigma**3
    Gamma[2,1,1] = a**2*np.sin(theta)*np.cos(theta)/(sigma*delta)
    Gamma[2,1,2] = Gamma[2,2,1] = r/sigma
    Gamma[2,2,2] = -a**2*np.sin(theta)*np.cos(theta)/sigma
    Gamma[2,3,3] = -(((r**2+a**2)**2-a**2*delta*np.sin(theta)**2)*sigma+2*self.M*(r**2+a**2)*a**2*r*np.sin(theta)**2)*np.sin(theta)*np.cos(theta)/sigma**3
    Gamma[3,0,2] = Gamma[3,2,0] = -2*self.M*a*r/(np.tan(theta)*sigma**2)
    Gamma[3,1,3] = Gamma[3,3,1] = (r*sigma**2+self.M*(a**4*np.sin(theta)**2*np.cos(theta)**2-r**2*(sigma+r**2+a**2)))/(sigma**2*delta)
    Gamma[3,2,3] = Gamma[3,3,2] = (sigma**2+2*self.M*a**2*r*np.sin(theta)**2)/(np.tan(theta)*sigma**2)

    return np.ascontiguousarray(Gamma)

@njit
def coord_pos(self, x):
    T0, X0, Y0, Z0 = x
    trans = np.array([T0, X0-self.pos.x, Y0-self.pos.y, Z0-self.pos.z], dtype=np.float64)
    trans = np.ascontiguousarray(trans)
    T, X, Y, Z = self.rot @ trans
    a = self.J / self.M
    s = X**2 + Y**2 + Z**2 - a**2

    out = np.zeros(4, dtype=np.float64)
    out[0] = T
    out[1] = r = np.sqrt((s + np.sqrt(s**2+4*a**2*Z**2)) / 2)
    out[2] = np.acos(Z/r)
    out[3] = np.atan2(Y,X)
    return np.ascontiguousarray(out)

@njit
def mink_pos(self, x):
    T, R, THETA, PHI = x
    a = self.J / self.M

    out = np.zeros(4, dtype=np.float64)
    out[0] = T
    out[1] = np.sqrt(R**2+a**2) * np.sin(THETA) * np.cos(PHI)
    out[2] = np.sqrt(R**2+a**2) * np.sin(THETA) * np.sin(PHI)
    out[3] = R * np.cos(THETA)
    out = np.ascontiguousarray(out)
    T0, X0, Y0, Z0 = self.rot_inv @ out
    return np.ascontiguousarray(np.array([T0, X0+self.pos.x, Y0+self.pos.y, Z0+self.pos.z], dtype=np.float64))

@njit
def jacobian(self, x):
    trans = x - np.array([0, self.pos.x, self.pos.y, self.pos.z])
    trans = np.ascontiguousarray(trans)
    T, X, Y, Z = self.rot @ trans
    a = self.J / self.M
    s = X**2 + Y**2 + Z**2 - a**2
    r = np.sqrt((s + np.sqrt(s**2+4*a**2*Z**2)) / 2)
    theta = np.acos(Z/r)

    J = np.zeros((4,4))
    J[0,0] = 1
    J[1,1] = j11 = X/(2*r) * (1+s/np.sqrt(s**2+4*a**2*Z**2))
    J[1,2] = j12 = Y/(2*r) * (1+s/np.sqrt(s**2+4*a**2*Z**2))
    J[1,3] = j13 = Z/(2*r) * (1+(s+2*a**2)/np.sqrt(s**2+4*a**2*Z**2))
    J[2,1] = Z/(r**2*np.sin(theta)) * j11
    J[2,2] = Z/(r**2*np.sin(theta)) * j12
    J[2,3] = (Z*j13-r)/(r**2*np.sin(theta))
    J[3,1] = -Y/(X**2+Y**2)
    J[3,2] = X/(X**2+Y**2)
    J = np.ascontiguousarray(J)
    return J @ self.rot

@njit
def jacobian_inv(self, x):
    T, R, THETA, PHI = x
    a = self.J / self.M
    J = np.zeros((4,4))
    J[0,0] = 1
    J[1,1] = R/np.sqrt(R**2+a**2) * np.sin(THETA) * np.cos(PHI)
    J[1,2] = np.sqrt(R**2+a**2) * np.cos(THETA) * np.cos(PHI)
    J[1,3] = -np.sqrt(R**2+a**2) * np.sin(THETA) * np.sin(PHI)
    J[2,1] = R/np.sqrt(R**2+a**2) * np.sin(THETA) * np.sin(PHI)
    J[2,2] = np.sqrt(R**2+a**2) * np.cos(THETA) * np.sin(PHI)
    J[2,3] = np.sqrt(R**2+a**2) * np.sin(THETA) * np.cos(PHI)
    J[3,1] = np.cos(THETA)
    J[3,2] = -R * np.sin(THETA)
    J = np.ascontiguousarray(J)
    return self.rot_inv @ J
