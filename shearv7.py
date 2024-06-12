"""
Dedalus script for Two fluid dust settling in a protoplanetary disc

"""
import numpy as np
import time
import h5py
from mpi4py import MPI
import dedalus.public as d3
import logging
import math
import os
import pathlib
logger = logging.getLogger(__name__)

#Aspect ratio 2
Lx, Lz = (1.0, 1.0)
nx, nz = (256,256)

# Create bases and domain
coords = d3.CartesianCoordinates('x', 'z')
dist   = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx),       dealias=3/2)
zbasis = d3.ChebyshevT(coords['z'],  size=nz, bounds=(-Lz/2, Lz/2), dealias=3/2)

# Set problem parameters
Re       = 2e-6
Sc       = 1.0
F        = Re*Sc
q        = 1.5
St0      = 0.1
Lambda   = 0.001
Lambda2  = Lambda*Lambda
bd0      = 0.1
eta      = 1e-2
St0      = St0/Lambda

# Fields
p  = dist.Field(name='p', bases=(xbasis,zbasis))
s  = dist.Field(name='s', bases=(xbasis,zbasis))
u  = dist.Field(name='u', bases=(xbasis,zbasis))
v  = dist.Field(name='v', bases=(xbasis,zbasis))
w  = dist.Field(name='w', bases=(xbasis,zbasis))
ud = dist.Field(name='ud', bases=(xbasis,zbasis))
vd = dist.Field(name='vd', bases=(xbasis,zbasis)) 
wd = dist.Field(name='wd', bases=(xbasis,zbasis))
bd = dist.Field(name='bd', bases=(xbasis,zbasis))
tau_p   = dist.Field(name='tau_p')
tau_s1  = dist.Field(name='tau_s1', bases=xbasis)
tau_s2  = dist.Field(name='tau_s2', bases=xbasis)
tau_u1  = dist.Field(name='tau_u1', bases=xbasis)
tau_u2  = dist.Field(name='tau_u2', bases=xbasis)
tau_v1  = dist.Field(name='tau_v1', bases=xbasis)
tau_v2  = dist.Field(name='tau_v2', bases=xbasis)
tau_w1  = dist.Field(name='tau_w1', bases=xbasis)
tau_w2  = dist.Field(name='tau_w2', bases=xbasis)
tau_ud1 = dist.Field(name='tau_ud1', bases=xbasis)
tau_ud2 = dist.Field(name='tau_ud2', bases=xbasis)
tau_vd1 = dist.Field(name='tau_vd1', bases=xbasis)
tau_vd2 = dist.Field(name='tau_vd2', bases=xbasis)
tau_wd1 = dist.Field(name='tau_wd1', bases=xbasis)
tau_wd2 = dist.Field(name='tau_wd2', bases=xbasis)
tau_bd1 = dist.Field(name='tau_bd1', bases=xbasis)
tau_bd2 = dist.Field(name='tau_bd2', bases=xbasis)

# First-order reductions
x, z       = dist.local_grids(xbasis, zbasis)
ex, ez     = coords.unit_vector_fields(dist)
dx         = lambda A: d3.Differentiate(A, coords['x'])
dz         = lambda A: d3.Differentiate(A, coords['z'])
lift_basis = zbasis.derivative_basis(1)
lift       = lambda A: d3.Lift(A, lift_basis, -1)
uz         = dz(u)  + lift(tau_u1) 
vz         = dz(v)  + lift(tau_v1) 
wz         = dz(w)  + lift(tau_w1) 
udz        = dz(ud) + lift(tau_ud1) 
vdz        = dz(vd) + lift(tau_vd1) 
wdz        = dz(wd) + lift(tau_wd1) 
bdz        = dz(bd) + lift(tau_bd1) 

#Gravity Field
gravity = dist.Field(name='gravity',bases=(xbasis,zbasis))
gravity['g'] = -Lambda**2*z

#Background density
logRho = dist.Field(name='log_rho',bases=(xbasis,zbasis))
logRho['g'] = -0.5*z**2

# Set problem variables
problem = d3.IVP([p,u,v,w,ud,vd,wd,bd,tau_u1,tau_u2,tau_v1,tau_v2,tau_w1,tau_w2,tau_ud1,tau_ud2,tau_vd1,tau_vd2,tau_wd1,tau_wd2,tau_bd1,tau_bd2,tau_p], namespace=locals())

# Incompressible Shearing-Box
#(Gas)
problem.add_equation("dx(u) + wz + tau_p = -w*dz(logRho)")
problem.add_equation("dt(u) + dx(p) - Re*(dx(dx(u)) + dz(uz)) + lift(tau_u2) = + 2*(v+eta)*Lambda            - u*dx(u) - w*uz + bd*(ud-u)/St0")
problem.add_equation("dt(v)         - Re*(dx(dx(v)) + dz(vz)) + lift(tau_v2) = - (2-q)*u*Lambda              - u*dx(v) - w*vz + bd*(vd-v)/St0")
problem.add_equation("dt(w) + dz(p) - Re*(dx(dx(w)) + dz(wz)) + lift(tau_w2) = -p*dz(logRho)   + gravity     - u*dx(w) - w*wz + bd*(wd-w)/St0")
#(Dust)
problem.add_equation("dt(bd) - F*(dx(dx(bd)) + dz(bdz)) + lift(tau_bd2) = - bd*(dx(ud) + wdz) - ud*dx(bd) - wd*bdz")
problem.add_equation("dt(ud) - Re*(dx(dx(ud)) + dz(udz)) + lift(tau_ud2) = + 2*vd*Lambda                   - ud*dx(ud) - wd*udz - (ud-u)/St0")
problem.add_equation("dt(vd) - Re*(dx(dx(vd)) + dz(vdz)) + lift(tau_vd2) = -(2-q)*ud*Lambda                - ud*dx(vd) - wd*vdz - (vd-v)/St0")
problem.add_equation("dt(wd) - Re*(dx(dx(wd)) + dz(wdz)) + lift(tau_wd2) = gravity                         - ud*dx(wd) - wd*wdz - (wd-w)/St0")

# Boundary conditions
#(Gas)
problem.add_equation("uz(z=-Lz/2)  = 0")
problem.add_equation("uz(z=+Lz/2)  = 0")
problem.add_equation("vz(z=-Lz/2)  = 0")
problem.add_equation("vz(z=+Lz/2)  = 0")
problem.add_equation("w(z=-Lz/2)   = 0")
problem.add_equation("w(z=+Lz/2)   = 0")
problem.add_equation("integ(p)     = 0")

problem.add_equation("udz(z=-Lz/2)  = 0")
problem.add_equation("udz(z=+Lz/2)  = 0")
problem.add_equation("vdz(z=-Lz/2)  = 0")
problem.add_equation("vdz(z=+Lz/2)  = 0")
problem.add_equation("wd(z=-Lz/2)   = 0")
problem.add_equation("wd(z=+Lz/2)   = 0")
problem.add_equation("bd(z=-Lz/2)   = 0")
problem.add_equation("bd(z=+Lz/2)   = 0")
#(Dust)

# Time-stepping
solver = problem.build_solver(d3.RK443)
solver.stop_sim_time = np.inf

#Restart Condition
if not pathlib.Path('restart.h5').exists():

    #Initial conditions
    #(Random perturbations, initialized globally for same results in parallel)
    amp    = 5e-3
    a      = 0.5
    kx     = 1.0
    kz     = 3.0

    #(Gas)
    u['g'] = (1 + amp*np.cos(2.0*np.pi*kx*x/Lx)*np.cos(2.0*np.pi*kz*z/Lz))*2*bd0*St0*np.exp(-0.5*z**2)*eta/((1+bd0)**2+St0**2)
    v['g'] = (1 + amp*np.cos(2.0*np.pi*kx*x/Lx)*np.cos(2.0*np.pi*kz*z/Lz))*(bd0*(1+bd0)/((1+bd0)**2+St0**2)-1)*eta 
    w['g'] = 0.0
    
    #(Dust)
    ud['g'] = -2*St0*eta/((1+bd0)**2 + St0**2)
    vd['g'] = -(1+bd0)*eta/((1+bd0)**2 + St0**2)
    wd['g'] = 0

    bd.fill_random('g', seed=42, distribution='normal', scale=0.1)
    bd.low_pass_filter(scales=0.5)
    bd['g'] += bd0*(1-0.99*np.tanh(abs(z/a)))

    dt = 0.005
    fh_mode = 'overwrite'

else:
    write, last_dt = solver.load_state('restart.h5', -1)
    dt = last_dt
    fh_mode = 'append'

#Integration parameters and CFL
solver.stop_sim_time  = 20000.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.02*Lx/nx
cfl        = d3.CFL(solver,initial_dt=0.01,safety=1.5,max_change=1.0,min_change=0.1,max_dt=1.0)
u_cfl      = dist.VectorField(coords, name='u_cfl', bases=(xbasis,zbasis))
u_cfl['g'][0] = u['g']
u_cfl['g'][1] = w['g']
cfl.add_velocity(u_cfl)

u_cfl['g'][0] = ud['g']
u_cfl['g'][1] = wd['g']
cfl.add_velocity(u_cfl)


# Analysis
snap = solver.evaluator.add_file_handler('snapshots', sim_dt=1.0, max_writes=1)
snap.add_task(bd,  name='bd')
snap.add_task(bdz, name='bdz')

snap.add_task(ud,  name='ud')
snap.add_task(udz, name='udz')

snap.add_task(vd,  name='vd')
snap.add_task(vdz, name='vdz')

snap.add_task(wd,  name='wd')
snap.add_task(wdz, name='wdz')

snap.add_task(p ,  name='p')

snap.add_task(u ,  name='u')
snap.add_task(uz , name='uz')

snap.add_task(v,   name='v')
snap.add_task(vz,  name='vz')

snap.add_task(w ,  name='w')
snap.add_task(wz , name='wz')

#Main loop
n = 1
logger.info('Starting loop')
start_time = time.time()
while solver.proceed:
    dt = cfl.compute_timestep()
    solver.step(dt)
    if solver.iteration % 10 == 0:
       logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
