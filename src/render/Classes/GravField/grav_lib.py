import sympy as sp
from itertools import product
from sympy.tensor.array import tensorcontraction, tensorproduct
from sympy.printing.numpy import NumPyPrinter
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def sympy_to_numba(expr, args, name, filename, extra_code=""):
    """Writes numba-decorated functions corresponding to expr in file."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    arg_str = ", ".join(str(arg) for arg in args)
    if extra_code != "":
        code = f"""
@njit
def {name}({arg_str}):
    {extra_code}
    return {NumPyPrinter().doprint(expr)}
"""
    else:
        code = f"""
@njit
def {name}({arg_str}):
    return {NumPyPrinter().doprint(expr)}
"""
    try:
        with open(filename, "a", encoding="utf-8") as file:
            file.write(code.strip() + "\n")
    except OSError as e: print(f"Error writing to file: {e}")


# Class (and subclasses) of gravitational fields 
class GravField:
    """Create a gravitational field.

    g: A sympy array of expressions describing the metric tensor in specified metric coordinates.

    coord_tf_sp: A sympy array of expressions describing a coordinate transformation from Minkowski to metric coordinates.

    mink_tf_sp: A sympy array of expressions describing a coordinate transformation from metric to Minkowski coordinates.
    
    Use the use() method to write the mathematical expressions of this metric tensor to executable Python functions."""

    def __init__(self, g:dict, coord_tf_sp:sp.Array, mink_tf_sp:sp.Array):
        x0, x1, x2, x3 = sp.symbols("x:4")
        t, x, y, z = sp.symbols("t x y z")
        self.coords = (x0, x1, x2, x3)
        self.mink = (t, x, y, z)
        self.coord_tf_sp = coord_tf_sp
        self.mink_tf_sp = mink_tf_sp

        # Evaluating sympy expressions of certain quantities
        self.metric_sp = g

        self.jacobian_sp = sp.MutableDenseNDimArray.zeros(4, 4)
        for (i,j) in product(range(4), repeat=2):
            self.jacobian_sp[i,j] = sp.simplify(sp.diff(coord_tf_sp[i], self.mink[j]))

        self.jacobian_inv_sp = sp.MutableDenseNDimArray.zeros(4, 4)
        for (i,j) in product(range(4), repeat=2):
            self.jacobian_inv_sp[i,j] = sp.simplify(sp.diff(mink_tf_sp[i], self.coords[j]))

        vt, vx, vy, vz = sp.symbols("vt vx vy vz")
        vel_mink = sp.Array([vt, vx, vy, vz])
        vel_metric = tensorcontraction(tensorproduct(self.jacobian_sp, vel_mink), (1,2))
        norm = sp.collect(tensorcontraction(tensorproduct(g, vel_metric, vel_metric), (0,2),(1,3)), vt)
        self.null_conds = sp.Array([norm.coeff(vt, 2), norm.coeff(vt, 1), norm.coeff(vt, 0)])
        print("Finished evaluating required gravitational expressions")

    def use(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("""# This is an automatically generated file. Do not change its content manually.
from numba import njit
import numpy

""")
        vx, vy, vz = sp.symbols("vx vy vz")
        for i in range(4): sympy_to_numba(self.coord_tf_sp[i], self.mink, f"X{i}", filename)
        for i in range(4): sympy_to_numba(self.mink_tf_sp[i], self.coords, ["T","X","Y","Z"][i], filename)
        for i in range(4):
            for j in range(i,4): sympy_to_numba(self.metric_sp[i,j], self.coords, f"g{i}{j}", filename)
        for (i,j) in product(range(4),repeat=2): sympy_to_numba(self.jacobian_sp[i,j], self.mink, f"J{i}{j}", filename)
        for (i,j) in product(range(4),repeat=2): sympy_to_numba(self.jacobian_inv_sp[i,j], self.coords, f"Jinv{i}{j}", filename)
        for i in range(3): sympy_to_numba(self.null_conds[i], tuple(list(self.mink)+[vx,vy,vz]), f"null_cond_{["a","b","c"][i]}", filename, extra_code="x0, x1, x2, x3 = X0(t, x, y, z), X1(t, x, y, z), X2(t, x, y, z), X3(t, x, y, z)")
        print("Finished writing expressions to file\n")
    

class Minkowski(GravField):
    """Describe the gravitational field with the Minkowski metric, i.e. no curvature, no mass, no energy."""
    def __init__(self):
        x0, x1, x2, x3 = sp.symbols("x:4")
        t, x, y, z = sp.symbols("t x y z")
        self.coords = (x0, x1, x2, x3)
        self.mink = (t, x, y, z)

        self.coord_tf_sp = sp.Array([t, x, y, z])
        self.mink_tf_sp = sp.Array([x0, x1, x2, x3])
        g = sp.Array([[-1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        super().__init__(g, self.coord_tf_sp, self.mink_tf_sp)

class Schwarzschild(GravField):
    """Describe the gravitational field with the Schwarzschild metric, i.e. the field of a single static point mass.
    
    X, Y, Z: The coordinates of the center of the point mass (black hole).
    
    M: The mass of the point mass (black hole) in natural coordinates (2*M is its Schwarzschild radius)."""
    
    def __init__(self, X:float, Y:float, Z:float, M:float):
        if M < 0: raise ValueError("M must be positive.")

        x0, x1, x2, x3 = sp.symbols("x:4")
        t, x, y, z = sp.symbols("t x y z")
        self.coords = (x0, x1, x2, x3)
        self.mink = (t, x, y, z)

        self.coord_tf_sp = sp.Array([t, sp.sqrt((x-X)**2+(y-Y)**2+(z-Z)**2), sp.acos((z-Z)/sp.sqrt((x-X)**2+(y-Y)**2+(z-Z)**2)), sp.atan2(y-Y, x-X)])
        self.mink_tf_sp = sp.Array([x0, x1*sp.sin(x2)*sp.cos(x3)+X, x1*sp.sin(x2)*sp.sin(x3)+Y, x1*sp.cos(x2)+Z])
        g = sp.Array([[-(1-2*M/x1), 0, 0, 0],
                      [0, 1/(1-2*M/x1), 0, 0],
                      [0, 0, x1**2, 0],
                      [0, 0, 0, x1**2*sp.sin(x2)**2]])
        super().__init__(g, self.coord_tf_sp, self.mink_tf_sp)
        
class Kerr(GravField):
    """Describe the gravitational field with the Kerr metric, i.e. the field of a rotating point mass.
    
    X, Y, Z: The coordinates of the center of the point mass (black hole).
    
    M: The mass of the point mass (black hole) in natural coordinates.
    
    J: The angular momentum of the point mass (black hole) in natural coordinates."""

    def __init__(self, X:float, Y:float, Z:float, M:float, J:float):
        if M < 0 or J < 0: raise ValueError("M and J must be positive.")
        if J > M: raise ValueError("J cannot be larger than M - a naked singularity will form.")

        x0, x1, x2, x3 = sp.symbols("x:4")
        t, x, y, z = sp.symbols("t x y z")
        self.coords = (x0, x1, x2, x3)
        self.mink = (t, x, y, z)

        a = J / M
        sigma = (x-X)**2 + (y-Y)**2 + (z-Z)**2 + a**2
        r = sp.sqrt((sigma - a**2 + sp.sqrt((sigma - a**2)**2 + 4*a**2*(z-Z)**2)) / 2)

        self.coord_tf_sp = sp.Array([t, r, sp.acos((z-Z)/r), sp.atan2(y-Y, x-X)])
        self.mink_tf_sp = sp.Array([x0, sp.sqrt(x1**2+a**2)*sp.sin(x2)*sp.cos(x3)+X, sp.sqrt(x1**2+a**2)*sp.sin(x2)*sp.sin(x3)+Y, x1*sp.cos(x2)+Z])

        rho = x1**2 + a**2*sp.cos(x2)**2
        delta = x1**2 - 2*M*x1 + a**2
        g = sp.Array([[-(1-2*M*x1/rho), 0, 0, -2*M*a*x1*sp.sin(x2)**2/rho],
                      [0, rho/delta, 0, 0],
                      [0, 0, rho, 0],
                      [-2*M*a*x1*sp.sin(x2)**2/rho, 0, 0, (x1**2+a**2+2*M*a**2*x1*sp.sin(x2)**2/rho)*sp.sin(x2)**2]])
        super().__init__(g, self.coord_tf_sp, self.mink_tf_sp)

