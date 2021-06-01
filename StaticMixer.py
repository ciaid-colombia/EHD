"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
											div(u) = 0
"""

from __future__ import print_function
from ufl import *
from fenics import *
from mshr import *
import mshr
import os
import dolfin
import numpy as np
from numpy import array, zeros, ones, any, arange, isnan, cos, arccos
import ctypes, ctypes.util, numpy, scipy.sparse, scipy.sparse.linalg, collections
import matplotlib.pyplot as plt
import time
import sys

###############################################################################
###############################################################################
############################ Own functions ####################################

#Computation time
start = time.time()
iterTime = start

# get file name
fileName = os.path.splitext(__file__)[0]

# VTK
vtkfile_u = File('%s.results/Velocity.pvd' % (fileName))
vtkfile_p = File('%s.results/Pressure.pvd' % (fileName))
vtkfile_a = File('%s.results/ConcenA.pvd' % (fileName))
vtkfile_b = File('%s.results/ConcenB.pvd' % (fileName))
vtkfile_r = File('%s.results/Density.pvd' % (fileName))
vtkfile_m = File('%s.results/Viscosity.pvd' % (fileName))
vtkfile_l = File('%s.results/Heating.pvd' % (fileName))

TimeMethod = ''

T = 1.0           	# final time
dt = 1e-8 	 	# time step size (perfectly working 1e-2)
t_steady = 1.0
num_steps = int(T/dt) 	# number of time steps
k  = Constant(dt)

#Geometric parameters and mesh
#InletInjectionLength = 0.026   #For 5% V/V 
#InletInjectionLength = 0.055  #For 10% V/V
#InletInjectionLength = 0.088  #For 15% V/V
InletInjectionLength = 0.166  #For 25% V/V
resolution = 500

#Fluid's parameters
Umean = 0.01399
mu_A = 0.00133        	# JetA1 dynamic viscosity
mu_B = 0.003786        	# Biodiesel dynamic viscosity
rho_A = 809.49            	# JetA1 density
rho_B = 841.26            	# Biodiesel density
LHV_A = 42.9            	# JetA1 density
LHV_B = 37.8            	# Biodiesel density
f  = Constant((0, 0))
f3  = Constant(0.0)
f_B = Constant(0)
Beta = Constant(1.0)          # 0 for Picard - 1 for Newton-Raphson 
C_D = Constant(7.4e-16)
psi_J = Constant(1) 
M_J = Constant(0.186)
Temperature = Constant(298.15)
V_B = Constant(0.05181068)

#VMS
C1 = Constant(12.0)
C2 = Constant(2.0)
C3 = Constant(1.0)
C4 = Constant(1.0)

############################################################################
############################ Mesh definition ###########################

## Create mesh for the 0 degrees
channel = mshr.Rectangle(Point(0.0, -0.5),Point(10, 0.5)) 
bafle_inf = mshr.Rectangle(Point(0.0, -0.5),Point(0.05, 0.0))
bafle_sup = mshr.Rectangle(Point(0.95, 0.0),Point(1.0, 0.5))
arreglo_bafle = bafle_inf+bafle_sup
arreglo1 = mshr.CSGTranslation(arreglo_bafle,Point(2.0, 0))
arreglo2 = mshr.CSGTranslation(arreglo1,Point(1.8, 0))
arreglo3 = mshr.CSGTranslation(arreglo2,Point(1.8, 0))
arreglo4 = mshr.CSGTranslation(arreglo3,Point(1.8, 0))
arreglos = arreglo1 + arreglo2 + arreglo3 + arreglo4

domain = channel - arreglos
mesh = mshr.generate_mesh(domain, resolution)

mfile = File("%s.results/mesh.pvd" % (fileName))
mfile << mesh

## Create mesh for the 30 degrees
bafle_inf_30 = mshr.Rectangle(Point(0.0, -0.55),Point(0.05,0.5/cos(pi/6)-0.5 ))
bafle_sup_30 = mshr.Rectangle(Point(0.95,0.5-0.5/cos(pi/6)),Point(1.0, 0.55))
rot_30_inf = mshr.CSGRotation(bafle_inf_30,Point(0.025, -0.5, 0), -pi/6)
rot_30_sup = mshr.CSGRotation(bafle_sup_30,Point(0.975, 0.5, 0), pi/6)
arreglo_bafle_30 = rot_30_inf+rot_30_sup
arreglo1_30 = mshr.CSGTranslation(arreglo_bafle_30,Point(2.0, 0))
arreglo2_30 = mshr.CSGTranslation(arreglo1_30,Point(1.8, 0))
arreglo3_30 = mshr.CSGTranslation(arreglo2_30,Point(1.8, 0))
arreglo4_30 = mshr.CSGTranslation(arreglo3_30,Point(1.8, 0))
arreglos_30 = arreglo1_30 + arreglo2_30 + arreglo3_30 + arreglo4_30
domain_30 = channel - arreglos_30
mesh_30 = mshr.generate_mesh(domain_30, resolution)

mfile_30 = File("%s.results/mesh_30.pvd" % (fileName))
mfile_30 << mesh_30

## Create mesh for the 45 degrees
bafle_inf_45 = mshr.Rectangle(Point(0.0, -0.55),Point(0.05, 0.5/cos(pi/4)-0.5))
bafle_sup_45 = mshr.Rectangle(Point(0.95, 0.5-0.5/cos(pi/4)),Point(1.0, 0.55))
rot_45_inf = mshr.CSGRotation(bafle_inf_45,Point(0.025, -0.5, 0), -pi/4)
rot_45_sup = mshr.CSGRotation(bafle_sup_45,Point(0.975, 0.5, 0), pi/4)
arreglo_bafle_45 = rot_45_inf+rot_45_sup
arreglo1_45 = mshr.CSGTranslation(arreglo_bafle_45,Point(2.0, 0))
arreglo2_45 = mshr.CSGTranslation(arreglo1_45,Point(1.8, 0))
arreglo3_45 = mshr.CSGTranslation(arreglo2_45,Point(1.8, 0))
arreglo4_45 = mshr.CSGTranslation(arreglo3_45,Point(1.8, 0))
arreglos_45 = arreglo1_45 + arreglo2_45 + arreglo3_45 + arreglo4_45
domain_45 = channel - arreglos_45
mesh_45 = mshr.generate_mesh(domain_45, resolution)
mfile_45 = File("%s.results/mesh_45.pvd" % (fileName))
mfile_45 << mesh_45


print('Number of nodes:', mesh.num_vertices() )
print('Number of elements:', mesh.num_cells() )
print('Number of DOFs:',mesh.num_vertices()*3 )
print('Number of DOFs:',mesh.num_vertices()*3 )

############################################################################
############################ Boundary definition ###########################

class NoslipBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary 
 
class InflowBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and near(x[0], 0.0)

class UpInjectionBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and near(x[1], 0.5) and x[0]<=InletInjectionLength

class DownInjectionBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and near(x[1], -0.5) and x[0]<=InletInjectionLength

class OutflowBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and near(x[0], 10.0)

######################################################################################
############################### Boundary conditions ##################################
######################################################################################
# Fluid Boundaries
noslip 			= NoslipBoundary()
inflow 			= InflowBoundary() 
outflow  		= OutflowBoundary() 
upinjec                 = UpInjectionBoundary()
downjec                 = DownInjectionBoundary()
inflow_profile = Expression(('1.5 * U_bar * 4.0 * x[1] * (Height1 - x[1]) / (Height1*Height1) * (1-cos(pi/2.0 * time))/2.0','0.0'), Height1 = 1, time=0.0, U_bar=Umean, degree=3)


chanfunc 	= MeshFunction('size_t', mesh, mesh.topology().dim()-1)
chanfunc.set_all(0)
noslip.mark  (chanfunc, 1)
inflow.mark (chanfunc, 2)
outflow.mark(chanfunc, 3)
upinjec.mark (chanfunc, 4)
downjec.mark (chanfunc, 5)

######################################################################################

############################ Initial conditions ############################

class InitialCondition(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
        values[3] = 0.0
    def value_shape(self):
        return (4,)

ic=InitialCondition(degree =2)

############################################################################
######################## Tensor and Matrix operations ######################

# Define symmetric gradient
def epsilon(u):
	 return sym(nabla_grad(u))

def convective(u1,u2):
	return dot(u1, nabla_grad(u2))

def c_A(c_B):
	return 1.0-c_B

def ln_rel(c_A,c_B,q_A,q_B):
	return exp(c_A*ln(q_A)+c_B*ln(q_B))

def eps_rel(mu_M):
	return C_D*psi_J*M_J*T/(mu_M*V_B)

############################################################################
################################## Matrix ##################################

# Galerkin Termns: 
def MAT_Gh(u, v, p, q, c_B, w):  
	return 2.0*ln_rel(c_A(c_B),c_B,mu_A,mu_B)*inner(epsilon(u), epsilon(v))*dx  \
			  - p*div(v)*dx + q*div(u)*dx\
		          + eps_rel(ln_rel(c_A(c_B),c_B,mu_A,mu_B))*dot(grad(c_B),grad(w))*dx

# NonLinear Terms:
def MAT_NLh(u, u_n, v, Beta, c_B, w):
	return Beta * inner( ln_rel(c_A(c_B),c_B,rho_A,rho_B) * convective(u,u_n), v )*dx \
			   	+ inner( ln_rel(c_A(c_B),c_B,rho_A,rho_B) * convective(u_n,u), v )*dx \
                                + dot(u_n, nabla_grad(c_B)) * w * dx 

# Stabilization Terms:
def MAT_VMSh(u, u_n, v, p, q, c_B, w, Tau1, Tau2, Tau3):
	return  Tau1 * inner( ln_rel(c_A(c_B),c_B,rho_A,rho_B) *convective(u_n,v)+nabla_grad(q), ln_rel(c_A(c_B),c_B,rho_A,rho_B) * convective(u_n,u)+nabla_grad(p) )*dx \
		  	+ Tau2 * div(v)*div(u)*dx \
                        + Tau3 * dot(u_n, nabla_grad(w))* dot(u_n, nabla_grad(c_B))*dx 

# Transient Term:
def MAT_Th(u,u_n,v, c_B, w, k,Tau1, Tau3):
	MAT = ln_rel(c_A(c_B),c_B,rho_A,rho_B) / k *inner( v , u )*dx \
		 + ln_rel(c_A(c_B),c_B,rho_A,rho_B)  / k * Tau1 * inner( ln_rel(c_A(c_B),c_B,rho_A,rho_B)*convective(u_n,v), u )*dx \
		          + ln_rel(c_A(c_B),c_B,rho_A,rho_B)* Tau1 / k * inner( nabla_grad(q)        , u )*dx \
                 + 1.0 / k * c_B * w * dx + 1.0 / k * Tau3 * dot(u_n, nabla_grad(w)) * c_B * dx  
	return MAT

############################################################################
################################## RHS  ####################################

# Galerkin Termns: 
# v * f
def VECT_Gh(v, w, c_B, f, f_B):  
	return ln_rel(c_A(c_B),c_B,rho_A,rho_B)*inner( v,f )*dx+ w*f_B*dx

# NonLinear Terms:
def VECT_NLh(u_n, v, Beta,c_B):
	return Beta*inner( ln_rel(c_A(c_B),c_B,rho_A,rho_B) * convective(u_n,u_n), v)*dx

# Stabilization Terms:
def VECT_VMSh(u_n,v,q,c_B,w,Tau1,Tau3,f,f_B):
	return Tau1 * ln_rel(c_A(c_B),c_B,rho_A,rho_B) *ln_rel(c_A(c_B),c_B,rho_A,rho_B)*  inner( convective(u_n,v)  , f)*dx \
		  + Tau1 * ln_rel(c_A(c_B),c_B,rho_A,rho_B)* inner( nabla_grad(q)      , f)*dx \
		  + Tau2 * ln_rel(c_A(c_B),c_B,rho_A,rho_B)*div(v) * f3 *dx \
	+ Tau3 * dot(u_n, nabla_grad(w)) * (f_B) * dx 

# Transient Term:
def VECT_Th(u_0,u_n,v,c_0B,c_B,w,Tau1,Tau3,k):
	VECT = ln_rel(c_A(c_B),c_B,rho_A,rho_B) / k *  inner( v                    , u_0 )*dx \
		  + ln_rel(c_A(c_B),c_B,rho_A,rho_B) / k * Tau1 * inner( ln_rel(c_A(c_B),c_B,rho_A,rho_B)*convective(u_n,v), u_0 )*dx \
		            + ln_rel(c_A(c_B),c_B,rho_A,rho_B)* Tau1 / k * inner( nabla_grad(q)        , u_0 )*dx \
           + 1.0 / k *  w * c_0B *dx \
           + 1.0 / k * Tau3 * dot(u, nabla_grad(w)) * c_0B *dx 
	return VECT


############################################################################
########################### Build function space ###########################

P2 = VectorElement("P", mesh.ufl_cell(), 1) #Second order for velocity
P1 = FiniteElement("P", mesh.ufl_cell(), 1) #First order for pressure
TH = MixedElement([P2, P1, P1])
W = FunctionSpace(mesh, TH)   #Our functional space
Q = FunctionSpace(mesh, P1)   #Our functional space

# Define functions for solutions at previous and current time steps

# For linear system
U = Function(W)

bcs = list()
bcs.append(DirichletBC(W.sub(0), Constant((0, 0)),	chanfunc, 1))
bcs.append(DirichletBC(W.sub(0), Constant((Umean, 0)),	chanfunc, 2))
bcs.append(DirichletBC(W.sub(1), Constant(0.),		chanfunc, 3))
bcs.append(DirichletBC(W.sub(0), Constant((0, -Umean)),	chanfunc, 4))
bcs.append(DirichletBC(W.sub(0), Constant((0, Umean)),	chanfunc, 5))
bcs.append(DirichletBC(W.sub(2), Constant(0.),	chanfunc, 2))
bcs.append(DirichletBC(W.sub(2), Constant(1.),	chanfunc, 4))
bcs.append(DirichletBC(W.sub(2), Constant(1.),	chanfunc, 5))

# Non-linear and old solutions
U_n = Function(W)
U_n.assign(interpolate(ic,W))

U_0 = Function(W)
U_0.assign(interpolate(ic,W))

U_00 =Function(W)
U_00.assign(interpolate(ic,W))

# Define trial and test functions
(u,p,c_B) = split(U)
(v,q,w) = TestFunctions(W)

# Get sub-functions
u_n, p_n, c_nB = split(U_n)
u_0, p_0, c_0B = split(U_0)
u_00, p_00, c_00B = split(U_00)

# Define facet normal and mesh size
h = 2.0*Circumradius(mesh)

############################################################################
############# Calculate mesh dependent algorithmic parameters ##############
############################################################################

vnorm = sqrt(dot(u_n, u_n))

# VMS Stabilization terms 
Tau1 = C1*ln_rel(c_A(c_B),c_B,mu_A,mu_B)/h**2.0 + C2*ln_rel(c_A(c_B),c_B,rho_A,rho_B)*vnorm/h
Tau1 = 1.0 / Tau1
Tau2 = C3*ln_rel(c_A(c_B),c_B,mu_A,mu_B) + C4*ln_rel(c_A(c_B),c_B,rho_A,rho_B)*vnorm*h
Tau3 = C1*eps_rel(ln_rel(c_A(c_B),c_B,mu_A,mu_B))/h**2.0 + C2*vnorm/h
Tau3 = 1.0 / Tau3
############################################################################
############################################################################

a = MAT_Gh(u, v, p, q, c_B, w) \
  + MAT_NLh(u, u_n, v, Beta, c_B, w) \
  + MAT_VMSh(u, u_n, v, p, q, c_B, w, Tau1, Tau2, Tau3)

L = VECT_Gh(v, w, c_B, f, f_B) \
  + VECT_NLh(u_n, v, Beta, c_B) \
  + VECT_VMSh(u_n,v,q,c_B,w,Tau1,Tau3,f,f_B) 


if TimeMethod == 'BDF':
	a = a + MAT_Th(u,u_n,v,c_B, w, k,Tau1, Tau3)
	L = L + VECT_Th(u_0,u_n,v,c_0B,c_B,w,Tau1,Tau3,k)
elif TimeMethod == 'BDF2':
 	a = a + 3.0/2.0*MAT_Th(u,u_n,v, c_B, w, k,Tau1, Tau3) 
 	L = L + 4.0/2.0*VECT_Th(u_0,u_n,v,c_0B,c_B,w,Tau1,Tau3,k) \
  				- 1.0/2.0*VECT_Th(u_00,u_n,v,c_00B,c_B,w,Tau1,Tau3,k)	

F = a - L
t = 0.0

for i in range(num_steps):
	
	print('TimeStep = ', i, 'Time = ', t)
	t += dt 
	if t > t_steady: inflow_profile.time = t_steady
	else: inflow_profile.time = t

        # Solve variational problem for time step
        solve(F == 0, U, bcs)

        (u,p,c_B)=U.split(True)

        # Update solution time step
        if TimeMethod=='BDF2' and i>1:
        	U_00.assign(U_0)
        U_0.assign(U)

        if (i % 1 == 0 ):
	   #Output fields
	   u.rename("velocity", "velocity") ;vtkfile_u << u
	   p.rename("pressure", "pressure") ;vtkfile_p << p
           outc_A = project(c_A(c_B),Q)
           outc_R = project(ln_rel(c_A(c_B),c_B,rho_A,rho_B),Q)
           outc_M = project(ln_rel(c_A(c_B),c_B,mu_A,mu_B),Q)
           outc_L = project(ln_rel(c_A(c_B),c_B,LHV_A,LHV_B),Q)
	   outc_A.rename("concA", "concA")    ;vtkfile_a << outc_A
	   c_B.rename("concB", "concB")    ;vtkfile_b << c_B
	   outc_R.rename("density", "density")    ;vtkfile_r << outc_R
	   outc_M.rename("viscosity", "viscosity")    ;vtkfile_m << outc_M
	   outc_L.rename("heating", "heating")    ;vtkfile_l << outc_L
