"""
Script demonstrating the use of external potentials to generate density profiles at a wall and in a pore
"""
import numpy as np
from scipy.constants import Avogadro, Boltzmann
from surfpack.pcsaft import PC_SAFT
import matplotlib.pyplot as plt
from surfpack.external_potential import LennardJones93, SlitPore, Steele, HardWall
from surfpack import Grid, Profile, Geometry

def show():
    plt.xlabel(r'$z$ [Å]')
    plt.ylabel(r'$\rho$ [Å$^{-3}$')
    plt.show()

dft = PC_SAFT('C2')
T = 250
p = 11.61e5
rho_b = [(1 / dft.eos.specific_volume(T, p, [1], dft.eos.VAPPH)[0]) * Avogadro / 1e30] # bulk density (particles / Å^3)
grid = Grid(1000, Geometry.PLANAR, 30) # Grid for planar geometry, 1000 points, 30 Å long

sigma = dft.get_sigma(0) * 1e10
eps_div_k = dft.get_eps_div_k(0)

sigma_wall = 3
eps_div_k_wall = 30.0

# Interaction from LB combination rules
sigma = 0.5 * (sigma + sigma_wall)
eps_div_k = np.sqrt(eps_div_k * eps_div_k_wall)

lj93 = [LennardJones93(sigma, eps_div_k * Boltzmann)]

rho = Profile.from_potential(rho_b, T, grid, lj93)
plt.plot(grid.z, rho[0])
plt.title('Ideal gas at Lennard-Jones 9-3 wall')
show()

rho = dft.density_profile_wall(rho_b, T, grid, lj93, verbose=True)
Profile.save_list(rho, 'saved_profiles/wall_93.json')
# rho = Profile.load_file('saved_profiles/wall_93.json')
plt.plot(grid.z, rho[0])
plt.title('Lennard Jones 9-3 potential at wall')
show()

rho_s = 0.114
delta = 3.35
steele = [Steele(sigma, 2 * eps_div_k * Boltzmann, delta, rho_s)]

rho = dft.density_profile_wall(rho_b, T, grid, steele, verbose=True)
Profile.save_list(rho, 'saved_profiles/wall_steele.json')
# rho = Profile.load_file('saved_profiles/wall_steele.json')
plt.plot(grid.z, rho[0] * 2)
plt.title('Steele potential at wall')
show()

wall = HardWall(1, is_pore=False) # When is_pore is True, the wall is at (z0, +infty). When False, the wall is at (- infty, z0)
rho = dft.density_profile_wall(rho_b, T, grid, wall, verbose=True)
Profile.save_list(rho, 'saved_profiles/wall_hard.json')
# rho = Profile.load_file('saved_profiles/wall_hard.json')
plt.plot(grid.z, rho[0] * 2)
plt.title('Hard wall at 1 Å')
show()

lj_slitpore = [SlitPore(30, lj93[0])] # Slitpore takes the pore width and an initialized potential as arguments
rho = dft.density_profile_wall(rho_b, T, grid, lj_slitpore, verbose=True)
Profile.save_list(rho, 'saved_profiles/lj_slitpore.json')
# rho = Profile.load_file('saved_profiles/lj_slitpore.json')
plt.plot(grid.z, rho[0] * 2)
plt.title('Mie 9-3 slit pore with width 30 Å')
show()