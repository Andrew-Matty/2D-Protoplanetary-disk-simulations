"""
Dedalus script simulating a 2D periodic incompressible shear flow with a passive
tracer field for visualization. This script demonstrates solving a 2D periodic
initial value problem. It can be ran serially or in parallel, and uses the
built-in analysis framework to save data snapshots to HDF5 files. The
`plot_snapshots.py` script can be used to produce plots from the saved data.
The simulation should take about 10 cpu-minutes to run.

The initial flow is in the x-direction and depends only on z. The problem is
non-dimensionalized usign the shear-layer spacing and velocity jump, so the
resulting viscosity and tracer diffusivity are related to the Reynolds and
Schmidt numbers as:

    nu = 1 / Reynolds
    D = nu / Schmidt

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shear_flow.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = 2, 2
Nx, Nz = 128, 128
Reynolds = 5e5
Schmidt = 0.01
dealias = 3/2
stop_sim_time = 200
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
s = dist.Field(name='s', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
v = dist.VectorField(coords, name='v', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_s1 = dist.Field(name='tau_s1', bases=xbasis)
tau_s2 = dist.Field(name='tau_s2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
tau_v1 = dist.VectorField(coords, name='tau_v1', bases=xbasis)
tau_v2 = dist.VectorField(coords, name='tau_v2', bases=xbasis)

# Substitutions
nu = 1 / Reynolds
D = nu / Schmidt
ts = 1e-3
eps = 0.1
Lambda = 0.1
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_v = d3.grad(v) + ez*lift(tau_v1) # First-order reduction
grad_s = d3.grad(s) + ez*lift(tau_s1) # First-order reduction

#zderivative
dz = lambda A: d3.Differentiate(A, coords['z'])

#Gravity Field
gravity = dist.Field(name='gravity',bases=(xbasis,zbasis))
gravity['g'] = -Lambda**2*z

angular = dist.VectorField(coords, name='rotation', bases=(xbasis,zbasis))
angular['g'] = 1

#Background density
log_rho = dist.Field(name='log_rho',bases=(xbasis,zbasis))
log_rho['g'] = -0.5*z**2

# Problem
#Gas
problem = d3.IVP([u, v, s, p, tau_p, tau_s1, tau_s2, tau_u1, tau_u2, tau_v1, tau_v2], namespace=locals())
#problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("trace(grad_u) + tau_p = -v@grad(log_rho)")
problem.add_equation("dt(s) - D*div(grad_s)            + lift(tau_s2) = - v@grad(s) - s*div(v)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) + lift(tau_u2) = - u@grad(u) + s*(v-u)/0.01 + gravity*ez - p*dz(log_rho)*ez")
problem.add_equation("dt(v) - nu*div(grad_v)           + lift(tau_v2) = - v@grad(v) -   (v-u)/0.01 + gravity*ez")

#Gas BC
problem.add_equation("u(z=-Lz/2) = 0.0")
problem.add_equation("u(z=+Lz/2) = 0.0")
problem.add_equation("integ(p) = 0") # Pressure gauge

#Dust BC
#problem.add_equation("dz(s)(z=-Lz/2) = 0.0") # Dust-to-gas ratio
#problem.add_equation("dz(s)(z=+Lz/2) = 0.0") # Dust-to-gas ratio
problem.add_equation("s(z=-Lz/2) = 0.0") # Dust-to-gas ratio
problem.add_equation("s(z=+Lz/2) = 0.0") # Dust-to-gas ratio
problem.add_equation("v(z=-Lz/2) = 0.0")
problem.add_equation("v(z=+Lz/2) = 0.0")


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
# Background shear
u['g'][0] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
v['g'][0] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))

# Dust
s.fill_random('g', seed=42, distribution='normal', scale=0.01)
s.low_pass_filter(scales=0.5)
s['g'] += -np.tanh((z-0.1)/0.5) + np.tanh((z+0.1)/0.5)

# Add small vertical velocity perturbations localized to the shear layers
u['g'][1] = 0.0
v['g'][1] = 0.0

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10)
snapshots.add_task(s, name='tracer')
snapshots.add_task(p, name='pressure')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)
CFL.add_velocity(v)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u@ex)**2, name='w2')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_w = np.sqrt(flow.max('w2'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, timestep, max_w))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
