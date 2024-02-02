import matplotlib.pyplot as plt
import numpy as np
from surfpack.pcsaft import PC_SAFT
from surfpack import PlanarGrid, Profile
import pandas as pd

comps = 'NC6' # 'C2'
dft = PC_SAFT(comps)
coarse_grid = PlanarGrid(100, 90) # For faster computations
grid = PlanarGrid(1000, 90) # For accurate computations
z = [1]
T1_lst = np.linspace(200, 300, 10)[:-1]
T2_lst = np.linspace(290, 300, 5)
T_C2_lst = np.concatenate((T1_lst, T2_lst))
T_C6_lst = np.concatenate((np.linspace(350, 450, 11), np.linspace(460, 500, 9)))
T_dct = {'C2' : T_C2_lst, 'NC6' : T_C6_lst}
T_lst = T_dct[comps]
gamma_lst = np.empty_like(T_lst)

rho = None
for i, T in enumerate(T_lst):
    print(f'Computing for T = {T} K')
    try:
        rho = Profile.load_file(f'saved_profiles/{comps}_surf_tens_{T}.json')
    except FileNotFoundError:
        rho = dft.density_profile_tp(T, 1e5, z, coarse_grid, rho_0=rho, verbose=True) # First converge the coarse grid
        rho = Profile.lst_on_grid(rho, grid) # Move profile to refined grid
        rho = dft.density_profile_tp(T, 1e5, z, grid, rho_0=rho, verbose=True) # Converge on refined grid
        Profile.save_list(rho, f'saved_profiles/{comps}_surf_tens_{T}.json') # Save profile for later

    gamma_lst[i] = dft.surface_tension(rho, T) # Compute surface tension
    rho = Profile.lst_on_grid(rho, coarse_grid) # Translate to coarse grid, for use as initial guess on next iteration

plt.plot(T_lst, gamma_lst * 1e20, label=comps) # Convert from J / Å^2 to J / m^2
plt.legend()
plt.xlabel(r'$T$ [K]')
plt.ylabel(r'$\gamma$ [J m$^{-2}$]')
plt.savefig(f'{comps}_surf_tens.pdf')
plt.show()