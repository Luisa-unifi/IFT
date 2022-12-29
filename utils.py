'''
Support Python code to compute a finite number of coefficients of the stream 
solution for a given SDE system.
'''


import numpy as np
from sympy import *
from sympy.matrices import Matrix
from sympy.solvers.ode.systems import dsolve_system
import matplotlib.pyplot as plt

#Generation of ODEs system (18) in Theorem 3 for the three-coloured trees system (19)

x1,x2,x3=symbols('x1,x2,x3')
x,y,w,z=symbols('x,y,w,z')
#Jacobian matrix
J=np.array([[1,-2*x2-2*x3,-2*x3-2*x2],[-2*x1-2*x3,1,-2*x3-2*x1],[-2*x1-2*x2,-2*x2-2*x1,1]])
J = Matrix(J)
det=J.det()
inv=J.inv()
#partial derivative w.r.t. x
v=np.matrix([-1,0,0])
#odes system
rst=-1*J.inv()*v.T
print(rst)


#Generation of ODEs system (18) in Theorem 3 for the one dimensional system (17)

J=np.array([[diff(w*z**4+x**2-y**2+y,w),diff(w*z**4+x**2-y**2+y,y), diff(w*z**4+x**2-y**2+y,z)],[diff(-w**2*y+x*z+w,w),diff(-w**2*y+x*z+w,y),diff(-w**2*y+x*z+w,z)],[diff(-w**3*x*z**5+w**4*z**4-w**2*x**3*z+w**3*x**2+x**2*y*z**2-x**2*z**2+w**2-x*z-w,w),diff(-w**3*x*z**5+w**4*z**4-w**2*x**3*z+w**3*x**2+x**2*y*z**2-x**2*z**2+w**2-x*z-w,y),diff(-w**3*x*z**5+w**4*z**4-w**2*x**3*z+w**3*x**2+x**2*y*z**2-x**2*z**2+w**2-x*z-w,z)]])
v=np.matrix([diff(w*z**4+x**2-y**2+y,x),diff(-w**2*y+x*z+w,x),diff(-w**3*x*z**5+w**4*z**4-w**2*x**3*z+w**3*x**2+x**2*y*z**2-x**2*z**2+w**2-x*z-w,x)])
J = Matrix(J)
det=J.det()
inv=J.inv()
rst=-1*J.inv()*v.T
print(rst)


