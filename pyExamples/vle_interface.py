import matplotlib.pyplot as plt
from surfpack.pcsaft import PC_SAFT
from surfpack import PlanarGrid, Profile
from scipy.constants import Avogadro

comps = 'C2'
dft = PC_SAFT(comps)
Tc = dft.eos.critical([1])[0]
T = Tc - 50
x = [1]
p = 1e5
grid = PlanarGrid(100, 60) # Planar geometry, 60 Å wide domain, 100 points

rho = dft.density_profile_tp(T, p, x, grid) # Ignores input for pressure for single components (pressure is computed from vle)
Profile.save_list(rho, f'saved_profiles/vle_{comps}_{int(T)}.json') # Save profile for later
# rho = Profile.load_file(f'saved_profiles/vle_{comps}_{int(T)}.json') # Load previously computed profile

plt.plot(grid.z, rho[0])
plt.xlabel(r'$z$ [Å]')
plt.ylabel(r'$\rho$ [Å$^{-3}$]')
plt.show()

p, _ = dft.eos.dew_pressure(T, x)
vl, = dft.eos.specific_volume(T, p, x, dft.eos.LIQPH)
vg, = dft.eos.specific_volume(T, p, x, dft.eos.VAPPH)

rho_l = (1 / vl) * Avogadro / 1e30
rho_g = (1 / vg) * Avogadro / 1e30

rho = dft.density_profile_twophase([rho_g], [rho_l], T, grid, verbose=True)
plt.plot(grid.z, rho[0])
plt.xlabel(r'$z$ [Å]')
plt.ylabel(r'$\rho$ [Å$^{-3}$]')
plt.show()