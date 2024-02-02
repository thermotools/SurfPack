import matplotlib.pyplot as plt

from surfpack.solvers import SequentialSolver, GridRefiner
from surfpack import Profile, PlanarGrid
from surfpack.pcsaft import PC_SAFT
from scipy.constants import Avogadro

dft = PC_SAFT('NC6')
grid = PlanarGrid(100, 50) # Planar geometry, 100 points, 50 Å wide
T = 400

solver = SequentialSolver('NT') # This solver will be used to solve an NT-spec problem
# Now we need to add some solvers that will be called sequentially
solver.add_picard(1e-3, mixing_alpha=0.01) # Start with a very careful picard solver, terminate at tolerance 1e-3
solver.add_picard(1e-4, mixing_alpha=0.05) # Move to a more agressive picard solver
solver.add_anderson(1e-6, beta_mix=0.02) # Use anderson until 1e-6
solver.add_anderson(1e-10, beta_mix=0.05) # More agressive anderson for the final steps

rho = dft.density_profile_tp(T, 1e5, [1], grid, solver=solver, verbose=True) # Pass the solver to a method for use
plt.plot(grid.z, rho[0])
plt.show()

# We can sometimes get a speed improvement by gradually refining the grid (i.e. starting with a coarse grid,
# converging the density profile, then using the coarse profile as an initial guess for a finer grid).
# We do this by using a GridRefiner solver object.

# First we need to define a coarse (initial) and fine (final) grid
coarsegrid = PlanarGrid(100, 50)
finegrid = PlanarGrid(1000, 50)

# Often, two refining steps is enough, and we can use
print(solver.constraints)
gridref = GridRefiner.init_twostep(solver, coarsegrid, grid, tol=1e-6)
# Which will simply converge the profile on the first grid to the limit "tol", before converging on the refined grid
# to the limit set in the "solver" object

# If we want more intermediate steps (three in the example below), we use the interface
gridref = GridRefiner.init_nsteps(solver, coarsegrid, finegrid, 3, tol_limits=[1e-5, 1e-6])
# Which will first converge the coarse grid to 1e-5, the converge an intermediate grid (between coarsegrid and finegrid)
# to 1e-6, before finally converging finegrid to the limit set by the solver.
rho = dft.density_profile_tp(T, 1e5, [1], finegrid, solver=gridref, verbose=True) # Pass the solver to a method for use
plt.plot(finegrid.z, rho[0])
plt.show()
