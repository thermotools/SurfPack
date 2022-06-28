#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utility import weighted_densities_1D, differentials_1D, \
    weighted_densities_pc_saft_1D, differentials_pc_saft_1D
from weight_functions_sph import spherical_weights, spherical_pc_saft_weights
from weight_functions_cosine_sine import planar_cosine_sine_weights, planar_cosine_sine_pc_saft_weights
from weight_functions import planar_weights, planar_pc_saft_weights
from constants import Geometry
import matplotlib.pyplot as plt
import pyfftw as fftw
import scipy.fft as sfft
from scipy.special import spherical_jn


class Weights_system_mc(object):
    """
    Multicomponent planar weigths
    """

    def __init__(self, functional,
                 geometry,
                 dr: float,
                 R: np.ndarray,
                 N: int,
                 mask_conv_results=None,
                 ms=None,
                 plweights=None,
                 wd=None,
                 diff=None):
        """

        Args:
            functional: Functional
            dr (float): Grid spacing
            R (ndarray): Particle radius
            N (int): Grid size
            mask_conv_results: Mask to avoid divide by zero
            ms (np.ndarray): Mumber of monomers
            plweights: Class type,defaults to planar_weights
            wd: Class type,defaults to weighted_densities_1D
            diff: Class type,defaults to differentials_1D
        """
        self.functional = functional
        self.pl_weights = []
        self.comp_weighted_densities = []
        self.comp_differentials = []
        self.nc = np.shape(R)[0]
        if ms is None:
            ms = np.ones(self.nc)

        if plweights is None:
            if geometry == Geometry.PLANAR:
                if (True if mask_conv_results is None else
                    all(mask == False for mask in mask_conv_results)):
                    plweights=planar_cosine_sine_weights
                else:
                    plweights=planar_weights
            elif geometry == Geometry.SPHERICAL:
                plweights=spherical_weights

        if wd is None:
            wd=weighted_densities_1D
        if diff is None:
            diff=differentials_1D


        for i in range(self.nc):
            self.pl_weights.append(plweights(dr=dr, R=R[i], N=N))
            self.comp_weighted_densities.append(
                wd(N=N, R=R[i], ms=ms[i], mask_conv_results=mask_conv_results))
            self.comp_differentials.append(diff(N=N, R=R[i]))

        # Overall weighted densities
        self.weighted_densities = wd(N=N, R=0.5, ms=ms[0])  # Dummy R

    def convolutions(self, rho):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """
        self.weighted_densities.set_zero()
        for i in range(self.nc):
            self.pl_weights[i].convolutions(
                self.comp_weighted_densities[i], rho[i])
            self.weighted_densities += self.comp_weighted_densities[i]
        self.weighted_densities.update_utility_variables()

    def convolution_n3(self, rho):
        """

            Args:
                rho (array_like): Density profile

            Returns:

            """
        self.weighted_densities.set_zero()
        for i in range(self.nc):
            self.pl_weights[i].convolution_n3(
                self.comp_weighted_densities[i], rho[i])
            self.weighted_densities += self.comp_weighted_densities[i]
        # self.weighted_densities.update_after_convolution()

    def correlation_convolution(self):
        """
        Calculate functional differentials and perform convolutions with the
        appropriate weight functions.
        """
        self.functional.differentials(self.weighted_densities)
        for i in range(self.nc):
            self.comp_differentials[i].set_functional_differentials(
                self.functional, i)
            self.pl_weights[i].correlation_convolution(
                self.comp_differentials[i])


class Weights_system_mc_pc_saft(Weights_system_mc):
    """
    Multicomponent planar weigths including PC-SAFT dispersion
    """

    def __init__(self, functional,
                 geometry,
                 dr: float,
                 R: np.ndarray,
                 N: int,
                 pcsaft: object,
                 mask_conv_results=None,
                 plweights=None,
                 wd=None,
                 diff=None):
        """

        Args:
            functional: Functional
            dr (float): Grid spacing
            R (np.ndarray): Particle radius
            N (int): Grid size
            pcsaft (pyctp.pcsaft): Thermopack object
            mask_conv_results: Mask to avoid divide by zero
            plweights: Class type,defaults to planar_pc_saft_weights
            wd: Class type,defaults to weighted_densities_pc_saft_1D
            diff: Class type,defaults to differentials_pc_saft_1D
        """

        self.thermo = pcsaft

        if plweights is None:
            if geometry == Geometry.PLANAR:
                if (True if mask_conv_results is None else
                    all(mask == False for mask in mask_conv_results)):
                    plweights=planar_cosine_sine_pc_saft_weights
                else:
                    plweights=planar_pc_saft_weights
            elif geometry == Geometry.SPHERICAL:
                plweights=spherical_pc_saft_weights

        if wd is None:
            wd=weighted_densities_pc_saft_1D
        if diff is None:
            diff=differentials_pc_saft_1D

        Weights_system_mc.__init__(self,
                                   functional,
                                   geometry,
                                   dr,
                                   R,
                                   N,
                                   mask_conv_results=mask_conv_results,
                                   ms=pcsaft.m,
                                   plweights=plweights,
                                   wd=wd,
                                   diff=diff)
        # Make sure rho_disp_array is allocated for mother density class
        self.weighted_densities.rho_disp_array = np.zeros((self.nc, N))

    def convolutions(self, rho):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """
        Weights_system_mc.convolutions(self, rho)
        for i in range(self.nc):
            self.weighted_densities.rho_disp_array[i, :] = \
                self.comp_weighted_densities[i].rho_disp[:]


if __name__ == "__main__":
    pass
