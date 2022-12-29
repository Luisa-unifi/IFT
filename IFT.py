'''
Python code to compute a finite number of coefficients of the stream solution for a given SDE 
system, using recurrence relation (24) in 'Implicit function theorem for the stream calculus'.

'''

import sympy
from sympy import *
import time
from collections import OrderedDict
init_printing(use_latex=False)
 
x, y, z, w, t , u, v, r= symbols('x y z w t u v r')
v1,v2, k, x1, x2, vx, vy= symbols('v1 v2 k x1 x2 vx vy')
a, b, c, d, e, f, g, h = symbols('a b c d e f g h')


def convolve_s(a, b):
    """
    Computes last element of convolution product a*b for equal length sequences a,b.
    """
    return sum([a[j] * b[len(a)-1 - j] for j in range(len(a))])

def prodsigma_s(s,t):
    '''
    Computes one element of the stream corresponding to the product of two terms.
    '''
    if (type(s)!=list):
        if (type(t)!=list):
            return s*t
        else:
            return s*t[-1]
    if type(t)!=list:
        return t*s[-1]  
    return convolve_s(s,t)

def sumsigma_s(s,t):
    '''
    Computes one element of the stream corresponding to the sum of two terms.
    '''
    if (type(s)!=list):
        if (type(t)!=list):
            return s+t
        if len(t)==1:
            return t[0]+s
        else:
            return t[-1]
    if (type(t)!=list):
        if len(s)==1:
            return s[0]+t
        else:
            return s[-1]
    return s[-1]+t[-1]


def topologicalInsert(e,dic,keyorder,s0):
    e=sympify(e)
    ar=e.args
    if isinstance(e,sympy.core.numbers.Number):
        dic[e]=e
        return keyorder,dic
    if ar==():
        return keyorder,dic
    for ei in ar:
        keyorder,dic=topologicalInsert(ei,dic,keyorder,s0)
    if not e in dic.keys():
        dic[e]=[e.subs(s0)]
        keyorder.append(e)
    return keyorder,dic
    
    
_root=sympy.Function("root")
def initDict(F,xlist,rho):
    '''Identifies all subterms appearing in the F-system and generates the corresponding 
    streams with only one element.'''
    s0={x:v for x,v in zip(xlist,rho)}
    dic={x:[v] for x,v in zip(xlist,rho)}
    keyorder=[]
    r0=_root(*F)
    keyorder,dic=topologicalInsert(r0,dic,keyorder,s0)
    keyorder.remove(r0)
    return keyorder,dic


def updateDict(F,xlist,dic,keyorder,rangevar=None,first=False):
    '''Generates a new element for the streams stored in the dic dictionary, 
    associated with the terms appearing in the F system '''
    if rangevar==None:
        rangevar=xlist
    
    newx=[0]*len(F)
    for xi,e,i in zip(xlist,F,range(len(F))):    # compute update streams of x1,...,xn
        if (xi in rangevar):       
            newx[i]=(dic[e][-1])
        elif first:
            newx[i]=(dic[e])
        else:
            newx[i]=0
    
    for xi,i in zip(xlist,range(len(F))):     # commit updates for xi
        dic[xi].append(newx[i])
    
    for e in keyorder:           # update rest of dictionary
        f=e.func
        ar=e.args    
        sl=[dic[ei] for ei in ar]
        if f==sympy.core.add.Add:
            dic[e].append(sumsigma_s(sl[0],sl[1]))
        elif f==sympy.core.mul.Mul:
            dic[e].append(prodsigma_s(sl[0],sl[1]))
        else:
            print('Error')
            return None
    return dic

def iterateStream(F,xlist,rho,N):
    '''
    Computes first N=10 coefficients of the unique stream solution sigma of E s.t. sigma(0)=r0
    
    Args:
        - F: system of stream differential equations.
        - xlist: variables appearing in system F.
        - rho: initial condition for system F.
        - N: number of coefficients to compute.

    Returns:
        - dic: dictionary that assigns to each subterm occurring in the polynomials of the right-hand part 
          of sysstem F, the corresponding stream consisting of N elements.
    '''
    start_time = time.time()
    F=[binarify(horner(e,xlist)) for e in F]
    ko,dic=initDict(F,xlist,rho)
    rangevar=[xlist[i] for i in range(len(F)) if not isinstance(sympify(F[i]),sympy.core.numbers.Number)]
    if N>=1:
        dic=updateDict(F,xlist,dic,ko,rangevar,first=True)
    for i in range(N-1):        
        dic=updateDict(F,xlist,dic,ko,rangevar,first=False)
    return dic    

def binarify(e):
    'reduces products with more than two terms in binary product'
    e=sympify(e)
    f=e.func
    ar=e.args
    if f==sympy.core.power.Pow:
        f=sympy.core.mul.Mul
        ar=(ar[0],)*ar[1]
    if ar==():
        return e
    e1=binarify(ar[0])
    if len(ar)==1:
        return f(e1,evaluate=False)
    if len(ar)==2:
        e2=binarify(ar[1])
        return f(e1,e2,evaluate=False)
    g=binarify(f(*ar[1:]))
    return f(e1,g, evaluate=False)


#--------- Utility: build SDE system from polynomial system ---------#

def convjac(E,xlist,rho): # only variables <>x must be listed in xlist! rho contains initial values for var. in xlist
    D=[]
    for xi in xlist:
        D.append(var('d'+str(xi)))
    DF=[sdm(f,[1]+D,[0]+rho,[x]+xlist)/1 for f in E]
    J=Matrix([[Poly(p,D).coeff_monomial(dxi) for dxi in D+[1]] for p in DF])
    return J[:,:-1], J[:,-1], D 

iDelta=var('iDelta')    
def poly2SDE(E,xlist,rho):              
    """ 
    Converts a system of polynomial equations into a system of stream differential equations (SDEs), a list of variables
    and initial conditions. 
    
    Args:
        - E: system of polynomial equations.
        - xlist: variables appearing in system E.
        - rho: initial condition for system E.

    Returns:
        a system of stream differential equations (SDEs), its variables and initial conditions.
    """
    s0={xi:v for xi,v in zip(xlist,rho)}
    s0[x]=0
    Jy,Jx,D=convjac(E,xlist,rho)
    adj=Jy.adjugate()
    det=Jy.det()
    S=list(-iDelta*adj@Jx)
    S_sub={dxi:e for dxi,e in zip(D,S)}
    ddet=(sdm(det,[1]+D,[0]+rho,[x]+xlist)/1).subs(S_sub)
    idet_eq=-(Rational(1/det.subs(s0)))*ddet*iDelta
    S=[1,idet_eq]+S
    return S, [x,iDelta]+xlist, [0,Rational(1/det.subs(s0))]+rho 



#--------------- for stream derivative ----------------------#
def conv(f,g,n):
    return sum([f**i*g**(n-i) for i in range(n+1)])


def sdm0(m,F,rho,Xlist):
    '''Computes stream derivative of monomial m, considering system F'''
    if m==():
        return 0
    if m[0]==0:
        sdm0(m[1:],F[1:],rho[1:],Xlist[1:])
    if Xlist[1:]==[]:
        tau=1
    else:
        tau=Poly({m[1:]:1},Xlist[1:])/1
    A=F[0]*tau*conv(Xlist[0],rho[0],m[0]-1)
    B=(rho[0]**m[0])*sdm0(m[1:],F[1:],rho[1:],Xlist[1:])
    return Poly(A+B,Xlist)/1
    
    
def sdm(p,F,rho,Xlist):
    '''Computes stream derivative of polyonimial p, considering system F'''
    if type(p)!=type(Poly(0,Xlist)):
        p=Poly(p,Xlist)
    d=p.as_dict()
    s=Poly(sum([sdm0(m,F,rho,Xlist)*d[m] for m in d.keys()]),Xlist)
    return s


###############################################################################
################################# Experiments #################################
###############################################################################

#three-coloured trees system (19)
'''
E = [y-x-(z+w)**2,z-(w+y)**2,w-(y+z)**2]   # E = system of polynomial equations
F,xlist,rho=poly2SDE([y-x-(z+w)**2,z-(w+y)**2,w-(y+z)**2],[w,y,z],[0,0,0])  # convert E into system of SDE F, plus list of variables and initial conditions. NB: 2nd input only mentions vars. of E. 3rd argument is r0, initial condition

for i in range (1,1000,100):
    start_time=time.time()
    res=iterateStream(F,xlist,rho,N=i)  # computes first N=10 coefficients of the unique stream solution sigma of E s.t. sigma(0)=r0
    print(time.time() - start_time)
    res[y]
'''

#one dimensional system (17)
'''
E = [w*z**4+x**2-y**2+y, -w**2*y+x*z+w,-w**3*x*z**5+w**4*z**4-w**2*x**3*z+w**3*x**2+x**2*y*z**2-x**2*z**2+w**2-x*z-w]
F,xlist,rho=poly2SDE([w*z**4+x**2-y**2+y,-w**2*y+x*z+w,-w**3*x*z**5+w**4*z**4-w**2*x**3*z+w**3*x**2+x**2*y*z**2-x**2*z**2+w**2-x*z-w],[w,y,z],[1,1,1])  
for i in range (0,500,10):
    start_time=time.time()
    res=iterateStream(F,xlist,rho,N=i) 
    print(time.time() - start_time)
    res[y]
'''



