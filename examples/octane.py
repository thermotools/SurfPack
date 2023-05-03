# Run script for the cDFT code that interfaces with Thermopack

import numpy as np
import sys
from thermopack.pcsaft import pcsaft
from thermopack.thermopack_state import Equilibrium
from src.interface import PlanarInterface
from src.constants import LenghtUnit, NA
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_palette('Dark2')
sns.set_style('ticks')

# Set up thermopack and equilibrium state
thermopack = pcsaft()
thermopack.init("NC8")
T_star = 0.7
T = 279.0
vle = Equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

# Define interface with initial tanh density profile
interf = PlanarInterface.from_tanh_profile(vle, thermopack.critical_temperature(1), domain_size=100.0, n_grid=1024, invert_states=True)

# Solve for equilibrium profile
interf.solve(log_iter=True)


s_E = interf.get_excess_entropy_density(reduced=False)

fig,ax = plt.subplots()
rho = interf.profile.rho_mix/(NA*interf.functional.grid_reducing_lenght**3)
ax.plot(interf.grid.z, s_E/rho, color="tab:blue")
ax.set_ylabel(r"$s^{\rm{E}}$ (J/mol/K)", color="tab:blue")
ax.set_xlabel("$z$ (Å)")
ax2=ax.twinx()
ax2.plot(interf.grid.z, 1.0e-3*rho, color="orange", ls="--")
ax2.set_ylabel(r"$\rho$ (kmol/m$^3$)", color="orange")
#leg = plt.legend(loc="best", numpoints=1, frameon=False)
ax.text(65, -60, 'Vapor-phase', fontsize=14)
ax.text(5, -60, 'Liquid-phase', fontsize=14)
ax.text(65, 20, 'Interesting peak', fontsize=12)
#ax.arrow(80, 16, -10, 0, head_width=2.0, head_length=2.0, fc='k', ec='k')
ax.set_xlim([0, 100.0])

plt.tight_layout()
plt.savefig("octane_279.pdf")
plt.show()
