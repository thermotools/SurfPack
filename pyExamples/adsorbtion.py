import matplotlib.pyplot as plt
import numpy as np
from surfpack.pcsaft import PC_SAFT

comps = 'NC6,C2'
comp1, comp2 = comps.split(',')
dft = PC_SAFT(comps)
dft.set_cache_dir('saved_profiles') # Computed profiles are stored in saved_profiles/
# See the docstring on Functional.py::profilecaching

# We can compute adsorbtion isotherms as so:
T = 300
gamma, x_liq = dft.adsorbtion_isotherm(T, n_points=10) # Specifying number of points = 10

plt.plot(x_liq, gamma[0] * 1e20, label=comp1) # Conversion from mol / Å^2 to mol / m^2
plt.plot(x_liq, gamma[1] * 1e20, label=comp2)
plt.xlabel(r'$x_{' + comp1 + '}$')
plt.ylabel(r'$\gamma$ [mol m$^{-2}$]')
plt.legend()
plt.show()

# If we want not only the liquid composition, but the vapour composition and pressure as well,
# we do
gamma, lve = dft.adsorbtion_isotherm(T, n_points=10, calc_lve=True)
print(lve) # This is a ThermoPack BinaryXY object (holding x, y and p)
plt.plot(lve[2] / 1e5, gamma[0] * 1e20, label=comp1) # Conversion from mol / Å^2 to mol / m^2
plt.plot(lve[2] / 1e5, gamma[1] * 1e20, label=comp2) # And conversion from Pa to bar
plt.xlabel(r'$p$ [bar]')
plt.ylabel(r'$\gamma$ [mol m$^{-2}$]')
plt.legend()
plt.show()

# Note: The grid points will not be equally spaced, but slightly more dense close to the
# x = 0 and x = 1, this is because
# 1) It helps the solver work faster.
# 2) The curve is often steeper near the edges.
# We can also specify the points at which we want to compute the adsorbtion:

points = np.linspace(0.3, 0.7, 10)
gamma, x_liq = dft.adsorbtion_isotherm(T, n_points=points)
plt.plot(x_liq, gamma[0] * 1e20, label=comp1) # Conversion from mol / Å^2 to mol / m^2
plt.plot(x_liq, gamma[1] * 1e20, label=comp2)
plt.xlabel(r'$x_{' + comp1 + '}$')
plt.ylabel(r'$\gamma$ [mol m$^{-2}$]')
plt.legend()
plt.show()

# Or, we can directly specify the minimum and maximum composition we are interested in
gamma, x_liq = dft.adsorbtion_isotherm(T, n_points=10, x_min=0.2, x_max=0.4)

plt.plot(x_liq, gamma[0] * 1e20, label=comp1) # Conversion from mol / Å^2 to mol / m^2
plt.plot(x_liq, gamma[1] * 1e20, label=comp2)
plt.xlabel(r'$x_{' + comp1 + '}$')
plt.ylabel(r'$\gamma$ [mol m$^{-2}$]')
plt.legend()
plt.show()