# A place for testing out functions/methods that would not otherwise be directly called.

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
def g10(x0, x1, x2, x3):
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
def g20(x0, x1, x2, x3):
    return 0
@njit
def g21(x0, x1, x2, x3):
    return 0
@njit
def g22(x0, x1, x2, x3):
    return x1**2
@njit
def g23(x0, x1, x2, x3):
    return 0
@njit
def g30(x0, x1, x2, x3):
    return 0
@njit
def g31(x0, x1, x2, x3):
    return 0
@njit
def g32(x0, x1, x2, x3):
    return 0
@njit
def g33(x0, x1, x2, x3):
    return x1**2*numpy.sin(x2)**2
@njit
def Gamma000(x0, x1, x2, x3):
    return 0
@njit
def Gamma001(x0, x1, x2, x3):
    return 0.5/(x1*(1.0*x1 - 1.0))
@njit
def Gamma002(x0, x1, x2, x3):
    return 0
@njit
def Gamma003(x0, x1, x2, x3):
    return 0
@njit
def Gamma010(x0, x1, x2, x3):
    return 0.5/(x1*(1.0*x1 - 1.0))
@njit
def Gamma011(x0, x1, x2, x3):
    return 0
@njit
def Gamma012(x0, x1, x2, x3):
    return 0
@njit
def Gamma013(x0, x1, x2, x3):
    return 0
@njit
def Gamma020(x0, x1, x2, x3):
    return 0
@njit
def Gamma021(x0, x1, x2, x3):
    return 0
@njit
def Gamma022(x0, x1, x2, x3):
    return 0
@njit
def Gamma023(x0, x1, x2, x3):
    return 0
@njit
def Gamma030(x0, x1, x2, x3):
    return 0
@njit
def Gamma031(x0, x1, x2, x3):
    return 0
@njit
def Gamma032(x0, x1, x2, x3):
    return 0
@njit
def Gamma033(x0, x1, x2, x3):
    return 0
@njit
def Gamma100(x0, x1, x2, x3):
    return 0.5*(1.0*x1 - 1.0)/x1**3
@njit
def Gamma101(x0, x1, x2, x3):
    return 0
@njit
def Gamma102(x0, x1, x2, x3):
    return 0
@njit
def Gamma103(x0, x1, x2, x3):
    return 0
@njit
def Gamma110(x0, x1, x2, x3):
    return 0
@njit
def Gamma111(x0, x1, x2, x3):
    return -0.5*(1.0*x1 - 1.0)/(x1**3*(1 - 1.0/x1)**2)
@njit
def Gamma112(x0, x1, x2, x3):
    return 0
@njit
def Gamma113(x0, x1, x2, x3):
    return 0
@njit
def Gamma120(x0, x1, x2, x3):
    return 0
@njit
def Gamma121(x0, x1, x2, x3):
    return 0
@njit
def Gamma122(x0, x1, x2, x3):
    return 1.0 - 1.0*x1
@njit
def Gamma123(x0, x1, x2, x3):
    return 0
@njit
def Gamma130(x0, x1, x2, x3):
    return 0
@njit
def Gamma131(x0, x1, x2, x3):
    return 0
@njit
def Gamma132(x0, x1, x2, x3):
    return 0
@njit
def Gamma133(x0, x1, x2, x3):
    return -1.0*(1.0*x1 - 1.0)*numpy.sin(x2)**2
@njit
def Gamma200(x0, x1, x2, x3):
    return 0
@njit
def Gamma201(x0, x1, x2, x3):
    return 0
@njit
def Gamma202(x0, x1, x2, x3):
    return 0
@njit
def Gamma203(x0, x1, x2, x3):
    return 0
@njit
def Gamma210(x0, x1, x2, x3):
    return 0
@njit
def Gamma211(x0, x1, x2, x3):
    return 0
@njit
def Gamma212(x0, x1, x2, x3):
    return 1.0/x1
@njit
def Gamma213(x0, x1, x2, x3):
    return 0
@njit
def Gamma220(x0, x1, x2, x3):
    return 0
@njit
def Gamma221(x0, x1, x2, x3):
    return 1.0/x1
@njit
def Gamma222(x0, x1, x2, x3):
    return 0
@njit
def Gamma223(x0, x1, x2, x3):
    return 0
@njit
def Gamma230(x0, x1, x2, x3):
    return 0
@njit
def Gamma231(x0, x1, x2, x3):
    return 0
@njit
def Gamma232(x0, x1, x2, x3):
    return 0
@njit
def Gamma233(x0, x1, x2, x3):
    return -1.0*numpy.sin(x2)*numpy.cos(x2)
@njit
def Gamma300(x0, x1, x2, x3):
    return 0
@njit
def Gamma301(x0, x1, x2, x3):
    return 0
@njit
def Gamma302(x0, x1, x2, x3):
    return 0
@njit
def Gamma303(x0, x1, x2, x3):
    return 0
@njit
def Gamma310(x0, x1, x2, x3):
    return 0
@njit
def Gamma311(x0, x1, x2, x3):
    return 0
@njit
def Gamma312(x0, x1, x2, x3):
    return 0
@njit
def Gamma313(x0, x1, x2, x3):
    return 1.0/x1
@njit
def Gamma320(x0, x1, x2, x3):
    return 0
@njit
def Gamma321(x0, x1, x2, x3):
    return 0
@njit
def Gamma322(x0, x1, x2, x3):
    return 0
@njit
def Gamma323(x0, x1, x2, x3):
    return 1.0*numpy.cos(x2)/numpy.sin(x2)
@njit
def Gamma330(x0, x1, x2, x3):
    return 0
@njit
def Gamma331(x0, x1, x2, x3):
    return 1.0/x1
@njit
def Gamma332(x0, x1, x2, x3):
    return 1.0*numpy.cos(x2)/numpy.sin(x2)
@njit
def Gamma333(x0, x1, x2, x3):
    return 0
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
    return x*z/(numpy.sqrt(-z**2/(x**2 + y**2 + z**2) + 1)*(x**2 + y**2 + z**2)**(3/2))
@njit
def J22(t, x, y, z):
    return y*z/(numpy.sqrt(-z**2/(x**2 + y**2 + z**2) + 1)*(x**2 + y**2 + z**2)**(3/2))
@njit
def J23(t, x, y, z):
    return -(-z**2/(x**2 + y**2 + z**2)**(3/2) + 1/numpy.sqrt(x**2 + y**2 + z**2))/numpy.sqrt((x**2 + y**2)/(x**2 + y**2 + z**2))
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
def K(x0, x1, x2, x3):
    return 12.0/x1**6
@njit
def normed_eq(t, x, y, z, vt, vx, vy, vz):
    x0, x1, x2, x3 = X0(t, x, y, z), X1(t, x, y, z), X2(t, x, y, z), X3(t, x, y, z)
    return vt**2*(-1 + 1.0/x1) + x1**2*(-vx*y/(x**2 + y**2) + vy*x/(x**2 + y**2))**2*numpy.sin(x2)**2 + x1**2*(vx*x*z/(numpy.sqrt(-z**2/(x**2 + y**2 + z**2) + 1)*(x**2 + y**2 + z**2)**(3/2)) + vy*y*z/(numpy.sqrt(-z**2/(x**2 + y**2 + z**2) + 1)*(x**2 + y**2 + z**2)**(3/2)) - vz*(-z**2/(x**2 + y**2 + z**2)**(3/2) + 1/numpy.sqrt(x**2 + y**2 + z**2))/numpy.sqrt(-z**2/(x**2 + y**2 + z**2) + 1))**2 + (vx*x/numpy.sqrt(x**2 + y**2 + z**2) + vy*y/numpy.sqrt(x**2 + y**2 + z**2) + vz*z/numpy.sqrt(x**2 + y**2 + z**2))**2/(1 - 1.0/x1)
@njit
def normed_diff_eq(t, x, y, z, vt, vx, vy, vz):
    x0, x1, x2, x3 = X0(t, x, y, z), X1(t, x, y, z), X2(t, x, y, z), X3(t, x, y, z)
    return 2*vt*(-1 + 1.0/x1)
