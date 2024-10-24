import matplotlib.pyplot as plt
import numpy as np

from surfpack.solvers import SequentialSolver, GridRefiner
from surfpack import Profile, SphericalGrid
from surfpack.LJspline import LJSpline_BH, LJSpline_WCA, LJSpline_UV
from scipy.constants import Avogadro

dft_bh = LJSpline_BH() # Initialize LJs - Barker-Henderson model
dft_uv = LJSpline_UV() # Initialize LJs - UV-model
dft_wca = LJSpline_WCA() # Initialize LJs - Weeks-Chandler-Anderson model
dft_dct = {'BH' : dft_bh, 'UV' : dft_uv, 'WCA' : dft_wca} # For easy iteration

def plot_rdf(ljs_type):
    dft = dft_dct[ljs_type]
    T_red = 2.5
    rho_red = 0.5

    sigma = dft.get_sigma()
    eps_div_k = dft.get_eps_div_k()

    rho = rho_red / (1e30 * sigma**3) # NOTE: Densities in SurfPack are (particles / Å^3)
    sigma_aa = sigma * 1e10 # Sigma in (Å)
    T = T_red * eps_div_k

    grid = SphericalGrid(500, 5 * sigma * 1e10) # Spherical geometry, 500 points, 5 sigma radius
    rdf = dft.radial_distribution_functions([rho], T, grid=grid)[0]

    # The RDF is returned as a "Profile" which we can treat as a callable:
    print(f'Using {ljs_type:<3} model: g(sigma) = {rdf(sigma_aa)}') # NOTE AGAIN: Lengths in SurfPack are in Angstrom

    plt.plot(rdf.grid.z / sigma_aa, rdf, label=ljs_type)

for k in dft_dct.keys():
    plot_rdf(k)

plt.legend()
plt.xlabel(r'$r$ ($\sigma$)')
plt.ylabel(r'$g(r)$ (-)')
plt.show()