#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import Geometry
from pyctp import pcsaft
from interface import Interface
import numpy as np


class Pore(Interface):
    """

    """

    def __init__(self,
                 geometry,
                 thermopack,
                 temperature,
                 external_potential,
                 domain_size=100.0,
                 n_grid=1024,
                 functional_kwargs={}):
        """
        Object holding specifications for classical DFT problem.
        Reduced particle size assumed to be d=1.0, and all other sizes are relative to this scale.

        Args:
            bulk_densities (ndarray): Bulk fluid density ()
            particle_diameters (ndarray): Particle diameter
            wall (str): Wall type (HardWall, SlitHardWall, None)
            domain_length (float): Length of domain
            functional (str): Name of hard sphere functional: Rosenfeld, WhiteBear, WhiteBear Mark II, Default Rosenfeld
            grid (int) : Grid size
            temperature (float): Reduced temperature
            quadrature (str): Quadrature to use during integration
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """

        # Init of base class
        Interface.__init__(self,
                           geometry=geometry,
                           thermopack=thermopack,
                           temperature=temperature,
                           domain_size=domain_size,
                           n_grid=n_grid,
                           functional_kwargs=functional_kwargs)
        self.v_ext = external_potential

class SlitPore(Pore):
    """

    """
    def __init__(self,
                 thermopack,
                 temperature,
                 external_potential,
                 domain_size=100.0,
                 n_grid=1024,
                 functional_kwargs={}):
        """
        Object holding specifications for classical DFT problem.
        Reduced particle size assumed to be d=1.0, and all other sizes are relative to this scale.

        Args:
            bulk_densities (ndarray): Bulk fluid density ()
            particle_diameters (ndarray): Particle diameter
            wall (str): Wall type (HardWall, SlitHardWall, None)
            domain_length (float): Length of domain
            functional (str): Name of hard sphere functional: Rosenfeld, WhiteBear, WhiteBear Mark II, Default Rosenfeld
            grid (int) : Grid size
            temperature (float): Reduced temperature
            quadrature (str): Quadrature to use during integration
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """

        # Init of base class
        Pore.__init__(self,
                      geometry=Geometry.PLANAR,
                      thermopack=thermopack,
                      temperature=temperature,
                      external_potential=external_potential,
                      domain_size=domain_size,
                      n_grid=n_grid,
                      functional_kwargs=functional_kwargs)


    @staticmethod
    def from_state(state,
                   external_potential,
                   domain_size=100.0,
                   n_grid=1024,
                   functional_kwargs={}):
        """
        Initialize tangens hyperbolicus profile

            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface. Default value 0.5.
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
        """
        pif = SlitPore(state.eos,
                       temperature=state.T,
                       external_potential=external_potential,
                       domain_size=domain_size,
                       n_grid=n_grid,
                       functional_kwargs=functional_kwargs)
        pif.constant_profile(state)
        return pif

class Surface(Pore):

    def __init__(self,
                 geometry,
                 functional,
                 external_potential,
                 surface_position,
                 domain_size=100.0,
                 n_grid=1024,
                 functional_kwargs={}):
        """
        Object holding specifications for classical DFT problem.
        Reduced particle size assumed to be d=1.0, and all other sizes are relative to this scale.

        Args:
            bulk_densities (ndarray): Bulk fluid density ()
            particle_diameters (ndarray): Particle diameter
            wall (str): Wall type (HardWall, SlitHardWall, None)
            domain_length (float): Length of domain
            functional (str): Name of hard sphere functional: Rosenfeld, WhiteBear, WhiteBear Mark II, Default Rosenfeld
            grid (int) : Grid size
            temperature (float): Reduced temperature
            quadrature (str): Quadrature to use during integration
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """

        # Init of base class
        Pore.__init__(self,
                      geometry=geometry,
                      functional=functional,
                      external_potential=external_potential,
                      domain_size=domain_size,
                      n_grid=n_grid,
                      functional_kwargs=functional_kwargs)


if __name__ == "__main__":
    pass
