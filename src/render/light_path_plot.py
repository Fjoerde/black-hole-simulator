import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from itertools import product
import inspect
from scipy.integrate import solve_ivp

t, r, theta, phi = sp.symbols("t r theta phi")
x = [t, r, theta, phi] # List of coordinates
G, c = 6.67e-11, 3e8 # Constants
M = 1e30 # Mass of object
Rs = 2*G*M/c**2 # Schwarzschild radius
g = sp.Matrix(np.array([  # Schwarzschild metric
    [(1-Rs/r)*c**2, 0, 0, 0], 
    [0, -1/(1-Rs/r), 0, 0],
    [0, 0, -r**2, 0],
    [0, 0, 0, -r**2*(sp.sin(theta))**2]]))

# Functions #######################################################
def lambdify(obj, idx_num): # Turn sympy expressions into functions for faster evaluation during simulation
    funcs = {}
    for i in product(range(4), repeat=idx_num):
        if type(obj) == sp.matrices.dense.MutableDenseMatrix:
            expr = obj[*i]
        elif inspect.isfunction(obj):
            expr = obj(*i)
        if expr == 0:
            continue
        funcs[i] = sp.lambdify(tuple(x), expr, "numpy")
    return funcs

def chr_sym(c, a, b): # Returns Christoffel symbol \Gamma^{c}_{ab}
    inv_g = g.inv()
    terms = [sp.diff(g[i,a],x[b]) + sp.diff(g[i,b],x[a]) - sp.diff(g[a,b],x[i]) for i in range(4)]
    return np.sum([inv_g[c,i]/2 * terms[i] for i in range(4)])

def geodesic_eq(t, y, R_esc): # For solving the differential equation
    x, v = y[:4], y[4:]
    A = np.zeros(4)
    for (c,a,b), func in chr_funcs.items():
        A[c] -= func(*x) * v[a] * v[b] # Geodesic Equation
    return np.concatenate([v, A])

# Event functions
def near_horizon(t, y, R_esc):
    return y[1] - 1.001*Rs
near_horizon.terminal = True
near_horizon.direction = -1

def esc_to_inf(t, y, R_esc):
    return y[1] - R_esc
esc_to_inf.terminal = True
esc_to_inf.direction = 1

# Function to simulation solution to geodesic equation
def light_path(X0, V0):
    path = solve_ivp(fun=geodesic_eq,
                     t_span=[0,1000],
                     y0=np.concatenate([X0, V0]),
                     method="RK45",
                     atol=1e-06, rtol=1e-8,
                     dense_output=True,
                     max_step=20.0,
                     events=[near_horizon, esc_to_inf],
                     args=(2.5*X0[1],))
    print(path.t_events)
    return path.y[:4].T

def get_init_cond(init_pos, init_vel): # init_pos and init_vel is in Cartesian, init_vel is a three-velocity,
                                       # Outputs list of initial four-position and four-velocities (normed to 0)
                                       # Outputs in Schwarzschild coordinates
    X0, V0 = [], []
    for i in range(len(init_pos)):
        xc, uc = init_pos[i], init_vel[i]
        t, x, y, z = xc
        ux, uy, uz = uc

        r = np.sqrt(x**2 + y**2 + z**2)
        theta, phi = np.arccos(z/r), np.arctan2(y,x)
        xp = np.array([t, r, theta, phi])

        vr = ux*np.sin(theta)*np.cos(phi) + uy*np.sin(theta)*np.sin(phi) + uz*np.cos(theta)
        vtheta = (ux*np.cos(theta)*np.cos(phi) + uy*np.cos(theta)*np.sin(phi) - uz*np.sin(theta)) / r
        vphi = (-ux*np.sin(phi) + uy*np.cos(phi)) / (r * np.sin(theta))
        Vt = sp.symbols("Vt")
        vp = np.array([Vt, vr, vtheta, vphi])
        norm = 0
        for (a,b), func in g_funcs.items():
            norm += func(*xp) * vp[a] * vp[b]
        vt = max(sp.solve(sp.Eq(norm, 0), Vt))
        vp[0] = vt
        
        X0.append(xp), V0.append(vp)
    return X0, V0


#################################################################

if __name__ == "__main__":
    # Lambdify metric tensor and Christoffel symbols
    g_funcs = lambdify(g, 2)
    chr_funcs = lambdify(chr_sym, 3)

    # Set initial conditions
    num, dim = 4, 2 # Number of light rays along one direction
    v_mag = 50 # Affects speed of simulation, does not affect trajectory
    init_pos = [np.array([0, -5e3, i, j]) for (i,j) in product(np.linspace(-6e3, 6e3, num),repeat=dim)] # In Cartesian coordinates
    init_vel = [np.array([v_mag, 0, 0]) for _ in range(len(init_pos))]

    # Simulate to find light ray paths
    Xi, Vi = get_init_cond(init_pos, init_vel)
    paths = [light_path(Xi[i], Vi[i]) for i in range(len(init_pos))]

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # For graphing out event horizon
    Theta, Phi = np.linspace(0, np.pi, 1000), np.linspace(0, 2*np.pi, 1000) 
    ax.plot_surface(Rs*np.outer(np.cos(Phi),np.sin(Theta)),
                    Rs*np.outer(np.sin(Phi),np.sin(Theta)),
                    Rs*np.outer(np.ones_like(Phi),np.cos(Theta)), color="k", edgecolor="none", alpha=0.35, zorder=4)
    
    # Light rays
    for i in range(len(paths)):
        x_plot = [j[1]*np.sin(j[2])*np.cos(j[3]) for j in paths[i]]
        y_plot = [j[1]*np.sin(j[2])*np.sin(j[3]) for j in paths[i]]
        z_plot = [j[1]*np.cos(j[2]) for j in paths[i]]
        ax.plot3D(x_plot, y_plot, z_plot, color="tab:olive", zorder=3)
        ax.plot3D(x_plot[0], y_plot[0], z_plot[0], "x", color="tab:blue", zorder=1)
        ax.plot3D(x_plot[-1], y_plot[-1], z_plot[-1], "x", color="tab:orange", zorder=2)
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    boundaries = []
    for i in [getattr(ax,f"get_{j}lim")() for j in "xyz"]:
        lb, ub = i
        boundaries += [np.abs(lb), np.abs(ub)]
    d = max(boundaries)
    ax.set_xlim(-d, d), ax.set_ylim(-d, d), ax.set_zlim(-d, d)
    ax.grid(False)
    ax.set_title("Light Rays Around a Black Hole")

    plt.tight_layout()
    plt.show()
