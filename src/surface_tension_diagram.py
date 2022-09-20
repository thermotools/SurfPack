#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
#from dft_numerics import dft_solver
from constants import Geometry, ProfileInit, LenghtUnit
from interface import PlanarInterface
import numpy as np
import matplotlib.pyplot as plt

class SurfaceTensionDiagram(object):
    """

    """

    def __init__(self,
                 curve,
                 geometry=Geometry.PLANAR,
                 domain_size=200.0,
                 n_grid=1024,
                 solver=None,
                 init_profiles=ProfileInit.TANH,
                 functional_kwargs={}):
        """Class for calculating surface tension along saturation curve

        Args:
            curve (phase_diagram): List of states to calculate surface tension
            geometry (int): PLANAR/POLAR/SPHERICAL
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            solver (dft_solver, optional): Solver for DFT
            init_profiles (ProfileInit): How to initialize profiles? Default: ProfileInit.TANH
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """
        self.surface_tension = []
        self.surface_tension_reduced = []

        #
        t_crit, _, _ = curve.vle_states[0].vapor.eos.critical(curve.vle_states[0].vapor.x)
        #if geometry == Geometry.PLANAR:
        for state in curve.vle_states:
            if (abs(t_crit - state.vapor.T) > 1.0e-4):
                interf = PlanarInterface.from_tanh_profile(state,
                                                           t_crit,
                                                           domain_size=domain_size,
                                                           n_grid=n_grid,
                                                           functional_kwargs=functional_kwargs)
                # Solve for equilibrium profile
                interf.solve(log_iter=False)
                # interf.plot_equilibrium_density_profiles(plot_reduced_densities=True,
                #                                          plot_equimolar_surface=True,
                #                                          grid_unit=LenghtUnit.REDUCED)
                st = interf.surface_tension()
                st_r = interf.surface_tension_real_units()
            else:
                st = 0.0
                st_r = 0.0
            self.surface_tension_reduced.append(st)
            self.surface_tension.append(st_r)

            # from_profile(vle,
            #          profile,
            #          domain_size=100.0,
            #          n_grid=1024,
            #          invert_states=False,
            #          functional_kwargs={})

        self.surface_tension_reduced = np.array(self.surface_tension_reduced)
        self.surface_tension = np.array(self.surface_tension)

if __name__ == "__main__":
    pass
