# This is an automatically generated file. Do not change its content manually.
from numba import njit
import numpy

@njit
def X0(t, x, y, z):
    return t
@njit
def X1(t, x, y, z):
    return numpy.sqrt(x**2 + y**2 + z**2)
@njit
def X2(t, x, y, z):
    return numpy.arccos(z/numpy.sqrt(x**2 + y**2 + z**2))
@njit
def X3(t, x, y, z):
    return numpy.arctan2(y, x)
@njit
def T(x0, x1, x2, x3):
    return x0
@njit
def X(x0, x1, x2, x3):
    return x1*numpy.sin(x2)*numpy.cos(x3)
@njit
def Y(x0, x1, x2, x3):
    return x1*numpy.sin(x2)*numpy.sin(x3)
@njit
def Z(x0, x1, x2, x3):
    return x1*numpy.cos(x2)
@njit
def g00(x0, x1, x2, x3):
    return -1 + 1.0/x1
@njit
def g01(x0, x1, x2, x3):
    return 0
@njit
def g02(x0, x1, x2, x3):
    return 0
@njit
def g03(x0, x1, x2, x3):
    return 0
@njit
def g11(x0, x1, x2, x3):
    return (1 - 1.0/x1)**(-1.0)
@njit
def g12(x0, x1, x2, x3):
    return 0
@njit
def g13(x0, x1, x2, x3):
    return 0
@njit
def g22(x0, x1, x2, x3):
    return x1**2
@njit
def g23(x0, x1, x2, x3):
    return 0
@njit
def g33(x0, x1, x2, x3):
    return x1**2*numpy.sin(x2)**2
@njit
def J00(t, x, y, z):
    return 1
@njit
def J01(t, x, y, z):
    return 0
@njit
def J02(t, x, y, z):
    return 0
@njit
def J03(t, x, y, z):
    return 0
@njit
def J10(t, x, y, z):
    return 0
@njit
def J11(t, x, y, z):
    return x/numpy.sqrt(x**2 + y**2 + z**2)
@njit
def J12(t, x, y, z):
    return y/numpy.sqrt(x**2 + y**2 + z**2)
@njit
def J13(t, x, y, z):
    return z/numpy.sqrt(x**2 + y**2 + z**2)
@njit
def J20(t, x, y, z):
    return 0
@njit
def J21(t, x, y, z):
    return x*z/(numpy.sqrt((x**2 + y**2)/(x**2 + y**2 + z**2))*(x**2 + y**2 + z**2)**(3/2))
@njit
def J22(t, x, y, z):
    return y*z/(numpy.sqrt((x**2 + y**2)/(x**2 + y**2 + z**2))*(x**2 + y**2 + z**2)**(3/2))
@njit
def J23(t, x, y, z):
    return (-x**2 - y**2)/(numpy.sqrt((x**2 + y**2)/(x**2 + y**2 + z**2))*(x**2 + y**2 + z**2)**(3/2))
@njit
def J30(t, x, y, z):
    return 0
@njit
def J31(t, x, y, z):
    return -y/(x**2 + y**2)
@njit
def J32(t, x, y, z):
    return x/(x**2 + y**2)
@njit
def J33(t, x, y, z):
    return 0
@njit
def Jinv00(x0, x1, x2, x3):
    return 1
@njit
def Jinv01(x0, x1, x2, x3):
    return 0
@njit
def Jinv02(x0, x1, x2, x3):
    return 0
@njit
def Jinv03(x0, x1, x2, x3):
    return 0
@njit
def Jinv10(x0, x1, x2, x3):
    return 0
@njit
def Jinv11(x0, x1, x2, x3):
    return numpy.sin(x2)*numpy.cos(x3)
@njit
def Jinv12(x0, x1, x2, x3):
    return x1*numpy.cos(x2)*numpy.cos(x3)
@njit
def Jinv13(x0, x1, x2, x3):
    return -x1*numpy.sin(x2)*numpy.sin(x3)
@njit
def Jinv20(x0, x1, x2, x3):
    return 0
@njit
def Jinv21(x0, x1, x2, x3):
    return numpy.sin(x2)*numpy.sin(x3)
@njit
def Jinv22(x0, x1, x2, x3):
    return x1*numpy.sin(x3)*numpy.cos(x2)
@njit
def Jinv23(x0, x1, x2, x3):
    return x1*numpy.sin(x2)*numpy.cos(x3)
@njit
def Jinv30(x0, x1, x2, x3):
    return 0
@njit
def Jinv31(x0, x1, x2, x3):
    return numpy.cos(x2)
@njit
def Jinv32(x0, x1, x2, x3):
    return -x1*numpy.sin(x2)
@njit
def Jinv33(x0, x1, x2, x3):
    return 0
@njit
def null_cond_a(t, x, y, z, vx, vy, vz):
    x0, x1, x2, x3 = X0(t, x, y, z), X1(t, x, y, z), X2(t, x, y, z), X3(t, x, y, z)
    return -1 + 1.0/x1
@njit
def null_cond_b(t, x, y, z, vx, vy, vz):
    x0, x1, x2, x3 = X0(t, x, y, z), X1(t, x, y, z), X2(t, x, y, z), X3(t, x, y, z)
    return 0
@njit
def null_cond_c(t, x, y, z, vx, vy, vz):
    x0, x1, x2, x3 = X0(t, x, y, z), X1(t, x, y, z), X2(t, x, y, z), X3(t, x, y, z)
    return x1**2*(-vx*y/(x**2 + y**2) + vy*x/(x**2 + y**2))**2*numpy.sin(x2)**2 + x1**2*(vx*x*z/(numpy.sqrt((x**2 + y**2)/(x**2 + y**2 + z**2))*(x**2 + y**2 + z**2)**(3/2)) + vy*y*z/(numpy.sqrt((x**2 + y**2)/(x**2 + y**2 + z**2))*(x**2 + y**2 + z**2)**(3/2)) + vz*(-x**2 - y**2)/(numpy.sqrt((x**2 + y**2)/(x**2 + y**2 + z**2))*(x**2 + y**2 + z**2)**(3/2)))**2 + (vx*x/numpy.sqrt(x**2 + y**2 + z**2) + vy*y/numpy.sqrt(x**2 + y**2 + z**2) + vz*z/numpy.sqrt(x**2 + y**2 + z**2))**2/(1 - 1.0/x1)
