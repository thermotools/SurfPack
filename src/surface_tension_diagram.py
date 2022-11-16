#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from dft_numerics import dft_solver
from constants import Geometry, ProfileInit, LenghtUnit, Specification
from interface import PlanarInterface, SphericalInterface
import numpy as np
import matplotlib.pyplot as plt
from pyctp.thermopack_state import meta_curve

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

class SphericalDiagram(object):
    """

    """

    def __init__(self,
                 vle,
                 initial_radius,
                 n_grid=1024,
                 solver=None,
                 calculate_bubble=True,
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
        self.interfaces = []

        # Solve for planar interface
        eos = vle.liquid.eos
        t_crit, _, _ = eos.critical(vle.liquid.x)
        self.planar_interf = PlanarInterface.from_tanh_profile(vle,
                                                               t_crit,
                                                               domain_size=200.0,
                                                               n_grid=n_grid,
                                                               functional_kwargs=functional_kwargs)
        self.planar_interf.solve()
        sigma0 = self.planar_interf.surface_tension_real_units()


        n = 25 # Number of meta-stable points
        self.phase = eos.LIQPH if calculate_bubble else eos.VAPPH
        self.z = vle.liquid.x if calculate_bubble else vle.vapor.x
        states = meta_curve.isothermal(eos, vle.temperature, self.z, n, self.phase)
        rd = initial_radius * 2.0
        # Define interface with initial tanh density profile
        spi = SphericalInterface.from_tanh_profile(vle,
                                                   t_crit,
                                                   radius=initial_radius,
                                                   domain_radius=rd,
                                                   n_grid=n_grid,
                                                   calculate_bubble=calculate_bubble,
                                                   sigma0=sigma0,
                                                   specification=Specification.CHEMICHAL_POTENTIAL,
                                                   functional_kwargs=functional_kwargs) #CHEMICHAL_POTENTIAL, NUMBER_OF_MOLES
        print("mul,mur",spi.bulk.left_state.chemical_potential(),spi.bulk.right_state.chemical_potential())

        if solver is None:
            solver=dft_solver()
            #.picard(tolerance=1.0e-5,max_iter=200,beta=0.05,ng_frequency=None).\
            #    anderson(mmax=50, beta=0.05, tolerance=1.0e-10,max_iter=200)

        spi.solve(solver=solver, log_iter=True)

        print("vl,vr",spi.bulk.left_state.v,spi.bulk.right_state.v)
        print("mul,mur",spi.bulk.left_state.chemical_potential(),spi.bulk.right_state.chemical_potential())
        # Solve for initial sphere
        for i, meta in enumerate(states.meta_states):
            if i == 0:
                continue
            dp = np.abs(meta.liquid.pressure() - meta.vapor.pressure()) + 0.001
            R = 2.0*sigma0/dp
            print("R",R*1e10)
            #print("v",meta.liquid.v)
            #!print("mu",meta.liquid.chemical_potential(),meta.vapor.chemical_potential())

        # if calculate_bubble:
        #     vz, rho = vle.vapor.eos.map_meta_isotherm(temperature=T,
        #                                               z=z,
        #                                               phase=PeTS.LIQPH,
        #                                               n=n)
        # else:
        #     # Calculate droplet

        # interf = SphericalInterface.from_tanh_profile(state,
        #                                               t_crit,
        #                                               domain_size=domain_size,
        #                                               n_grid=n_grid,
        #                                               functional_kwargs=functional_kwargs)

        # #if geometry == Geometry.PLANAR:
        # for state in curve.vle_states:
        #     if (abs(t_crit - state.vapor.T) > 1.0e-4):
        #         interf = SphericalInterface.from_tanh_profile(state,
        #                                                    t_crit,
        #                                                    domain_size=domain_size,
        #                                                    n_grid=n_grid,
        #                                                    functional_kwargs=functional_kwargs)
        #         # Solve for equilibrium profile
        #         interf.solve(log_iter=False)
        #         # interf.plot_equilibrium_density_profiles(plot_reduced_densities=True,
        #         #                                          plot_equimolar_surface=True,
        #         #                                          grid_unit=LenghtUnit.REDUCED)
        #         st = interf.surface_tension()
        #         st_r = interf.surface_tension_real_units()
        #     else:
        #         st = 0.0
        #         st_r = 0.0
        #     self.surface_tension_reduced.append(st)
        #     self.surface_tension.append(st_r)

        #     # from_profile(vle,
        #     #          profile,
        #     #          domain_size=100.0,
        #     #          n_grid=1024,
        #     #          invert_states=False,
        #     #          functional_kwargs={})

        # self.surface_tension_reduced = np.array(self.surface_tension_reduced)
        # self.surface_tension = np.array(self.surface_tension)

if __name__ == "__main__":
    pass
