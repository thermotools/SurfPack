#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import Geometry
from weight_functions import WeightFunctions, ConvType
from grid import Grid
import numpy as np
from scipy.fft import dct, idct, dst, idst, fft, ifft
import matplotlib.pyplot as plt

class WeightedDensities():
    """
    """

    def __init__(self, n_grid: int, wfs: WeightFunctions, ms: float):
        """

        Args:
            n_grid:
            wfs:
            ms:
        """
        self.n_grid = n_grid
        self.ms = ms
        self.wfs = wfs
        # Hard-coded FMT densities
        self.n0 = np.zeros(n_grid)
        self.n1 = np.zeros(n_grid)
        self.n2 = np.zeros(n_grid)
        self.n3 = np.zeros(n_grid)
        self.n1v = np.zeros(n_grid)
        self.n2v = np.zeros(n_grid)
        # Additional densities
        self.n = {}
        for wf in wfs.wfs:
            alias = wfs.wfs[wf].alias
            if alias not in wfs.fmt_aliases:
                self.n[alias] = np.zeros(n_grid)
            else:
                if "v" in alias:
                    if "1" in alias:
                        self.n[alias] = self.n1v
                    else:
                        self.n[alias] = self.n2v
                elif "0" in alias:
                    self.n[alias] = self.n0
                elif "1" in alias:
                    self.n[alias] = self.n1
                elif "2" in alias:
                    self.n[alias] = self.n2
                elif "3" in alias:
                    self.n[alias] = self.n3

        # Pointer to local densities
        self.rho = None

    def convolve_densities(self, rho_inf: float, frho_delta: np.ndarray, temperature_diff_convolution=False):
        """
        Args:
            rho_inf (float): Boundary density
            frho_delta (np.ndarray): Fourier transformed density profile
            temperature_diff_convolution (bool): Convolve for temperature differentials? Default False
        """
        # Loop all weight functions
        for wf in self.wfs:
            if temperature_diff_convolution:
                self.wfs[wf].convolve_densities_T(rho_inf,
                                                  frho_delta,
                                                  self.n[wf])
            else:
                self.wfs[wf].convolve_densities(rho_inf,
                                                frho_delta,
                                                self.n[wf])
                for wf in self.wfs:
                    self.wfs[wf].update_dependencies(self)

        # Account for segments
        if not temperature_diff_convolution:
            self.update_after_convolution()

        # print("n0",self.n["w0"])
        # print("n1",self.n["w1"])
        # print("n2",self.n["w2"])
        # print("n3",self.n["w3"])
        # print("nv1",self.n["wv1"])
        # print("nv2",self.n["wv2"])
        # print("n_disp",self.n["w_disp"])

    def convolve_densities_by_type(self, rho: np.ndarray, conv_type=ConvType.REGULAR):
        """
        Args:
            rho (np.ndarray): Density profile
            conv_type (ConvType): How to convolve
        """
        if conv_type==ConvType.REGULAR or conv_type==ConvType.REGULAR_T:
            rho_delta = np.zeros(self.grid.n_grid)
            rho_inf = rho[-1]
            rho_delta[:] = rho[:] - rho_inf
            if self.grid.geometry == Geometry.PLANAR:
                frho_delta = dct(rho_delta, type=2)
            elif self.grid.geometry == Geometry.SPHERICAL:
                frho_delta = dst(rho_delta*self.grid.z, type=2)
            elif self.grid.geometry == Geometry.POLAR:
                pass
            self.convolve_densities(rho_inf, frho_delta, conv_type==ConvType.REGULAR_T)
        else:
            # Loop all weight functions
            for wf in self.wfs:
                self.wfs[wf].convolve_densities_complex(rho, self.n[wf], conv_type)
            for wf in self.wfs:
                self.wfs[wf].update_dependencies(self)

            self.update_after_convolution()

    def update_after_convolution(self):
        """
        """
        # Account for segments
        for wd in self.n:
            if wd in self.wfs.fmt_aliases:
                self.n[wd][:] *= self.ms

    def __getitem__(self, wd):
        return self.n[wd]

    def reset(self, rho):
        """
        Set weights to zero
        """
        for wd in self.n:
            self.n[wd].fill(0.0)
        # Need access to local density in some functionals:
        self.rho = rho

    def __iadd__(self, other):
        """
        Add weights
        """
        # Add FMT densities
        for wd in self.n:
            if wd in self.wfs.fmt_aliases:
                self.n[wd][:] += other.n[wd][:]
        return self

    def set_testing_values(self, rho=None):
        """
        Set some dummy values for testing differentials
        """
        if rho is not None:
            self.n0[:] = np.sum(rho)
            self.n1[:] = np.sum(self.R * rho)
            self.n2[:] = 4 * np.pi * np.sum(self.R ** 2 * rho)
            self.n3[:] = 4 * np.pi * np.sum(self.R ** 3 * rho) / 3
            self.n1v[:] = - 1.0e-3*self.n0[:]
            self.n2v[:] = 4*np.pi*np.average(self.R)*self.n1v[:]
        else:
            self.n2[:] = 3.0
            self.n3[:] = 0.5
            self.n2v[:] = 6.0
            self.n0[:] = 1.0
            self.n1[:] = 2.0
            self.n1v[:] = 5.0

        self.N = 1
        self.update_utility_variables()

    def print(self, index=None):
        """

        Args:
            print_utilities (bool): Print also utility variables
        """
        print("\nWeighted densities:")
        for wd in self.n:
            if index is None:
                print(wd.replace("w", "n") +": ", self.n[wd][:])
            else:
                print(wd.replace("w", "n") +": ", self.n[wd][index])

    def plot(self, r, show=True):
        """

        Args:
            r (np.ndarray): Spatial resolution
            show (bool): Execute plt.show ?
            mask (ndarray of bool): What to plot?
        """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        for i, wd in enumerate(self.n):
            label = wd.replace("w", "n")
            if "wv" in wd:
                ax2.plot(r, self.n[wd][:], label=label, color=LCOLORS[i])
            else:
                ax1.plot(r, self.n[wd][:], label=label, color=LCOLORS[i])
        ax1.set_xlabel("$r$")
        ax1.set_ylabel("$n$")
        ax2.set_ylabel("$n_v$")
        if show:
            leg = ax1.legend(loc="best", numpoints=1)
            leg.get_frame().set_linewidth(0.0)
            leg = ax2.legend(loc="best", numpoints=1)
            leg.get_frame().set_linewidth(0.0)
            plt.show()
        return ax1, ax2

    def get_fmt_densities(self, index):
        """

        Args:
            index (int): Grid position
        Returns:
            np.ndarray: Array of FMT weighted densities
        """

        return np.array([self.n0[index],
                         self.n1[index],
                         self.n2[index],
                         self.n3[index],
                         self.n1v[index],
                         self.n2v[index]])

    def get_fmt_densities_grid(self):
        """
        Returns:
            np.ndarray: 2D Array of FMT weighted densities
        """
        return np.c_[self.n0, self.n1, self.n2, self.n3, self.n1v, self.n2v]

class WeightedDensitiesMaster(WeightedDensities):
    """
    """

    def __init__(self,
                 n_grid: int,
                 wfs_list: list[WeightFunctions],
                 ms: np.ndarray):
        """

        Args:
            n_grid:
            wfs:
            ms:
        """
        WeightedDensities.__init__(self, n_grid, wfs=wfs_list[0], ms=1.0)
        self.wfs_list = wfs_list
        self.nc = len(wfs_list)

        # Component weigthed densities
        self.comp_weighted_densities = []
        for i, wfs in enumerate(wfs_list):
            self.comp_weighted_densities.append(WeightedDensities(n_grid=n_grid, wfs=wfs, ms=ms[i]))

        # Utilities
        self.n3neg = np.zeros(n_grid)
        self.n3neg2 = np.zeros(n_grid)
        self.n2v2 = np.zeros(n_grid)
        self.logn3neg = np.zeros(n_grid)
        self.n32 = np.zeros(n_grid)

        # Additional densities as nc arrays
        for wf in wfs.wfs:
            alias = wfs.wfs[wf].alias
            if alias not in wfs.fmt_aliases:
                self.n[alias] = np.zeros((self.nc, n_grid))

    def convolve_densities(self, rho_inf: float, frho_delta: np.ndarray, i: int, temperature_diff_convolution=False):
        """
        Args:
            rho_inf (float): Boundary density
            frho_delta (np.ndarray): Fourier transformed density profile
            i (int): Component index
            temperature_diff_convolution (bool): Convolve for temperature differentials? Default False
        """
        self.comp_weighted_densities[i].convolve_densities(rho_inf, frho_delta, temperature_diff_convolution)
        # Add component contribution to overall density
        self += self.comp_weighted_densities[i]
        for wf in self.wfs_list[i]:
            alias = self.wfs_list[i].wfs[wf].alias
            if alias not in self.wfs_list[i].fmt_aliases:
                self.n[alias][i, :] = \
                    self.comp_weighted_densities[i].n[alias][:]

    def convolve_densities_by_type(self, rho: np.ndarray, i: int, conv_type=ConvType.REGULAR):
        """
        Args:
            rho (np.ndarray): Density profile
            i (int): Component index
        """
        self.comp_weighted_densities[i].convolve_densities_by_type(rho, conv_type)
        # Add component contribution to overall density
        self += self.comp_weighted_densities[i]
        for wf in self.wfs_list[i]:
            alias = self.wfs_list[i].wfs[wf].alias
            if alias not in self.wfs_list[i].fmt_aliases:
                self.n[alias][i, :] = \
                    self.comp_weighted_densities[i].n[alias][:]

    def update_utility_variables(self):
        """
        """
        self.n3neg[:] = 1.0 - self.n3[:]
        self.n3neg2[:] = self.n3neg[:] ** 2
        self.n2v2[:] = self.n2v[:] ** 2
        self.logn3neg[:] = np.log(self.n3neg[:])
        self.n32[:] = self.n3[:] ** 2

    def perturbate(self, alias, eps=1.0e-5, ic=0):
        """ Method intended for debugging functional differentials
        Args:
        alias (string): Name of weigthed density
        eps (float): Size and direction of relative perturbation
        ic (int): Component index
        Return:
        (np.ndarray): Density before perturbation
        """
        n = np.zeros_like(self.n3neg)
        ni = None
        if alias == "rho":
            n[:] = self.rho.densities[ic][:]
            self.rho.densities[ic][:] += self.rho.densities[ic][:]*eps
        elif alias in self.wfs.fmt_aliases:
            # Perturbate component density
            ni = np.zeros_like(self.n3neg)
            n[:] = self.n[alias][:]
            ni[:] = self.comp_weighted_densities[ic].n[alias][:]
            self.comp_weighted_densities[ic].n[alias][:] += ni[:]*eps
            self.n[alias][:] += ni[:]*eps
            self.update_utility_variables()
        else:
            n[:] = self.n[alias][ic,:]
            self.n[alias][ic,:] += self.n[alias][ic,:]*eps
        return n, ni

    def set_density(self, n, ni, alias, ic=0):
        """ Set weighted density
        Args:
        n (np.ndarray): Densities
        alias (string): Name of weigthed density
        ic (int): Component index
        """
        if alias == "rho":
            self.rho.densities[ic][:] = n[:]
        elif alias in self.wfs.fmt_aliases:
            self.n[alias][:] = n[:]
            self.comp_weighted_densities[ic].n[alias][:] = ni[:]
            self.update_utility_variables()
        else:
            self.n[alias][ic,:] = n[:]


class CompWeightedDifferentials():
    """
    """

    def __init__(self, n_grid: int, wfs: WeightFunctions, ms: float):
        """

        Args:
            N:
            R:
            mask_conv_results:
        """
        self.n_grid = n_grid
        self.ms = ms
        self.wfs = wfs
        # Hard coded FMT differentials
        self.d0 = np.zeros(n_grid)
        self.d1 = np.zeros(n_grid)
        self.d2 = np.zeros(n_grid)
        self.d3 = np.zeros(n_grid)
        self.d1v = np.zeros(n_grid)
        self.d2v = np.zeros(n_grid)

        # Convolved properties
        self.d_conv = {} # Effective results
        self.d_eff = np.zeros(n_grid)

        # Additional differentials
        self.d = {}
        for wf in wfs:
            alias = wfs[wf].alias
            self.d_conv[alias] = np.zeros(n_grid)
            if alias not in wfs.fmt_aliases:
                self.d[alias] = np.zeros(n_grid)
            else:
                if "v" in alias:
                    if "1" in alias:
                        self.d[alias] = self.d1v
                    else:
                        self.d[alias] = self.d2v
                elif "0" in alias:
                    self.d[alias] = self.d0
                elif "1" in alias:
                    self.d[alias] = self.d1
                elif "2" in alias:
                    self.d[alias] = self.d2
                elif "3" in alias:
                    self.d[alias] = self.d3

        # Differentials not requiring convolutions
        self.mu_of_rho = np.zeros(n_grid)

        # One - body direct correlation function
        self.corr = np.zeros(n_grid)

    def d_effective(self, wf):
        """

        Returns:
        """
        assert self.wfs[wf].convolve
        self.d_eff[:] = self.d[wf][:]
        for wfi in self.wfs:
            if not self.wfs[wfi].convolve and self.wfs[wfi].calc_from == wf:
                self.d_eff[:] += self.wfs[wfi].prefactor_evaluated*self.d[wfi][:]
        return self.d_eff

    def update_after_convolution(self):
        """

        Returns:
        """
        self.corr.fill(0.0)
        for wf in self.wfs:
            if self.wfs[wf].convolve:
                self.corr[:] -= self.d_conv[wf][:]
        # Add differentials with respect to local density
        self.corr[:] -= self.mu_of_rho[:]

    def set_functional_differentials(self, functional, ic):
        """

        Args:
            functional:
            ic:

        Returns:

        """
        self.d0[:] = self.ms*functional.d0[:, ic]
        self.d1[:] = self.ms*functional.d1[:, ic]
        self.d2[:] = self.ms*functional.d2[:, ic]
        self.d3[:] = self.ms*functional.d3[:, ic]
        self.d1v[:] = self.ms*functional.d1v[:, ic]
        self.d2v[:] = self.ms*functional.d2v[:, ic]

        for wf in self.wfs:
            alias = self.wfs[wf].alias
            if alias not in self.wfs.fmt_aliases:
                self.d[alias][:] = functional.diff[alias][:, ic]

        if functional.mu_of_rho is not None:
            self.mu_of_rho[:] = functional.mu_of_rho[:, ic]

    def print(self, index=None):
        """
        """
        print("\nDifferentials")
        for wd in self.d:
            if index is None:
                print(wd.replace("w", "d") +": ", self.d[wd][:])
            else:
                print(wd.replace("w", "d") +": ", self.d[wd][index])

    # def plot(self, r, show=True):
    #     """

    #     Args:
    #         r (np.ndarray): Spatial resolution
    #         show (bool): Execute plt.show ?
    #         mask (ndarray of bool): What to plot?
    #     """
    #     if mask is None:
    #         mask = np.full(len(r), True, dtype=bool)
    #     fig, ax1 = plt.subplots()
    #     ax1.plot(r[mask], self.d2eff_conv[mask], label="d2_sum", color="r")
    #     ax1.plot(r[mask], self.d3_conv[mask], label="d3", color="g")
    #     #ax1.plot(r[mask], self.n2[mask], label="n2", color="b")
    #     #ax1.plot(r[mask], self.n3[mask], label="n3", color="orange")
    #     ax2 = ax1.twinx()
    #     ax2.plot(r[mask], self.d2veff_conv[mask],
    #              label="d2v_sum", color="orange")
    #     #ax2.plot(r[mask], self.n1v[mask], label="n1v", color="cyan")
    #     ax1.set_xlabel("$r$")
    #     ax1.set_ylabel("$d$")
    #     ax2.set_ylabel("$d_v$")
    #     if show:
    #         ax1.plot(r[mask], self.corr[mask], label="corr", color="b")
    #         leg = ax1.legend(loc="best", numpoints=1)
    #         leg.get_frame().set_linewidth(0.0)
    #         leg = ax2.legend(loc="best", numpoints=1)
    #         leg.get_frame().set_linewidth(0.0)
    #         plt.show()
    #     return ax1, ax2

class Convolver(object):
    """

    """

    def __init__(self,
                 grid: Grid,
                 functional,
                 R: np.ndarray,
                 R_T: np.ndarray):
        """Class handeling convolution

        Args:
            grid (Grid): Grid class
            functional (Functional): DFT functional

        Returns:
            None
        """
        self.functional = functional
        self.grid = grid

        # Set up storage for weigted densities and differentials
        self.comp_differentials = []
        self.comp_wfs = []
        for i in range(functional.thermo.nc):
            self.comp_wfs.append(WeightFunctions.Copy(functional.wf))
            # Set up Fourier weights
            for wfun in self.comp_wfs[i]:
                self.comp_wfs[i][wfun].generate_fourier_weights(grid, R[i], R_T[i])
            self.comp_differentials.append(CompWeightedDifferentials(n_grid=grid.n_grid, wfs=self.comp_wfs[i], ms=functional.thermo.m[i]))

        # Overall weighted densities
        self.weighted_densities = WeightedDensitiesMaster(n_grid=grid.n_grid,
                                                          wfs_list=self.comp_wfs,
                                                          ms=functional.thermo.m)
        # Overall weighted densities used for temperature differentials
        self.weighted_densities_T = WeightedDensitiesMaster(n_grid=grid.n_grid,
                                                            wfs_list=self.comp_wfs,
                                                            ms=functional.thermo.m)

    def convolve_density_profile(self, rho):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """
        self.weighted_densities.reset(rho)
        rho_delta = np.zeros(self.grid.n_grid)
        for i in range(self.functional.thermo.nc):
            rho_inf = rho.densities[i][-1]
            rho_delta[:] = rho.densities[i][:] - rho_inf
            if self.grid.geometry == Geometry.PLANAR:
                frho_delta = dct(rho_delta, type=2)
            elif self.grid.geometry == Geometry.SPHERICAL:
                frho_delta = dst(rho_delta*self.grid.z, type=2)
            elif self.grid.geometry == Geometry.POLAR:
                pass
            self.weighted_densities.convolve_densities(rho_inf, frho_delta, i)
            # # Loop all weight functions
            # for wf in self.comp_wfs[i]:
            #     self.comp_wfs[i].wfs[wf].convolve_densities(rho_inf,
            #                                                 frho_delta,
            #                                                 self.weighted_densities.comp_weighted_densities[i].n[wf])
            # for wf in self.functional.wf.wfs:
            #     self.comp_wfs[i].wfs[wf].update_dependencies(self.weighted_densities.comp_weighted_densities[i])
            # # Account for segments
            # self.weighted_densities.comp_weighted_densities[i].update_after_convolution()
            # # print("n0",self.comp_weighted_densities[i].n["w0"])
            # # print("n1",self.comp_weighted_densities[i].n["w1"])
            # # print("n2",self.comp_weighted_densities[i].n["w2"])
            # # print("n3",self.comp_weighted_densities[i].n["w3"])
            # # print("nv1",self.comp_weighted_densities[i].n["wv1"])
            # # print("nv2",self.comp_weighted_densities[i].n["wv2"])
            # # print("n_disp",self.comp_weighted_densities[i].n["w_disp"])

            # print("xxx",i,self.weighted_densities.comp_weighted_densities[i].get_fmt_densities(0))

            # # Add component contribution to overall density
            # self.weighted_densities += self.weighted_densities.comp_weighted_densities[i]
            # print("yyy",i,self.weighted_densities.comp_weighted_densities[i].get_fmt_densities(0))
            # for wf in self.comp_wfs[i]:
            #     alias = self.comp_wfs[i].wfs[wf].alias
            #     if alias not in self.comp_wfs[i].fmt_aliases:
            #         self.weighted_densities.n[alias][i, :] = \
            #             self.weighted_densities.comp_weighted_densities[i].n[alias][:]
        self.weighted_densities.update_utility_variables()

        # Calculate differentials
        self.update_functional_differentials()
        for i in range(self.functional.nc):
            # Loop all weight functions
            for wf in self.comp_wfs[i]:
                if self.comp_wfs[i][wf].convolve:
                    self.comp_wfs[i][wf].convolve_differentials(self.comp_differentials[i].d_effective(wf),
                                                                self.comp_differentials[i].d_conv[wf])
            # Calculate one-particle correlation
            self.comp_differentials[i].update_after_convolution()

    def update_functional_differentials(self):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """
        # Calculate differentials
        self.functional.differentials(self.weighted_densities)
        for i in range(self.functional.nc):
            self.comp_differentials[i].set_functional_differentials(
                self.functional, i)

    def get_differential_sum(self, n_alpha: WeightedDensitiesMaster):
        """
        Perform convolutions for weighted densities

        Args:
            n_alpha (WeightedDensitiesMaster): Weigthed densities from another convolver instance
        Returns:
           f0 (np.ndarray): Differential evaluated with internal weigthed density multiplied with provided weighted densities (dfdn_0*n_alpha)
        """
        f0_na = np.zeros(self.grid.n_grid)
        for i in range(self.functional.nc):
            for wf in self.comp_wfs[i].wfs:
                alias = self.comp_wfs[i].wfs[wf].alias
                if alias in self.comp_wfs[i].fmt_aliases:
                    f0_na[:] += self.comp_differentials[i].d[alias][:]*n_alpha.n[alias][:]
                else:
                    f0_na[:] += self.comp_differentials[i].d[alias][:]*n_alpha.n[alias][i, :]
            #f0_na[:] += self.comp_differentials[i].mu_of_rho[:]*n_alpha.rho.densities[i][:]
        return f0_na

    def convolve_density_profile_T(self, rho):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """

        self.weighted_densities_T.reset(rho)
        rho_delta = np.zeros(self.grid.n_grid)
        for i in range(self.functional.thermo.nc):
            rho_inf = rho.densities[i][-1]
            rho_delta[:] = rho.densities[i][:] - rho_inf
            if self.grid.geometry == Geometry.PLANAR:
                frho_delta = dct(rho_delta, type=2)
            elif self.grid.geometry == Geometry.SPHERICAL:
                frho_delta = dst(rho_delta*self.grid.z, type=2)
            elif self.grid.geometry == Geometry.POLAR:
                pass
            self.weighted_densities_T.convolve_densities(rho_inf,
                                                         frho_delta,
                                                         i,
                                                         temperature_diff_convolution=True)
            # # Loop all weight functions
            # for wf in self.comp_wfs[i]:
            #     self.comp_wfs[i].wfs[wf].convolve_densities_T(rho_inf,
            #                                                   frho_delta,
            #                                                   self.comp_weighted_densities[i].n[wf])
            # # for wf in self.functional.wf.wfs:
            # #     self.comp_wfs[i].wfs[wf].update_dependencies(self.comp_weighted_densities[i])
            # # Account for segments
            # #self.comp_weighted_densities[i].update_after_convolution()
            # # print("n0",self.comp_weighted_densities[i].n["w0"])
            # # print("n1",self.comp_weighted_densities[i].n["w1"])
            # # print("n2",self.comp_weighted_densities[i].n["w2"])
            # # print("n3",self.comp_weighted_densities[i].n["w3"])
            # # print("nv1",self.comp_weighted_densities[i].n["wv1"])
            # # print("nv2",self.comp_weighted_densities[i].n["wv2"])
            # # print("n_disp",self.comp_weighted_densities[i].n["w_disp"])

            # # Add component contribution to overall density
            # self.weighted_densities_T += self.comp_weighted_densities[i]
            # for wf in self.comp_wfs[i]:
            #     alias = self.comp_wfs[i].wfs[wf].alias
            #     if alias not in self.comp_wfs[i].fmt_aliases:
            #         self.weighted_densities_T.n[alias][i, :] = \
            #             self.comp_weighted_densities[i].n[alias][:]

    def functional_temperature_differential_convolution(self, rho):
        """
        Get entropy per volume (J/m3/K)
        """

        f_T = np.zeros(self.grid.n_grid)
        # dndT
        self.convolve_density_profile_T(rho)
        # dfdn
        self.update_functional_differentials()
        # Sum dfdn*dndT
        for wf in self.comp_wfs[0]:
            if len(np.shape(self.weighted_densities_T.n[wf])) == 1:
                f_T[:] += self.comp_differentials[0].d[wf][:]*self.weighted_densities_T.n[wf][:]
            else:
                for i in range(self.functional.thermo.nc):
                    # Loop components
                    f_T[:] +=  self.comp_differentials[i].d[wf][:]*self.weighted_densities_T.n[wf][i, :]
        # dfdT
        dfdT = self.functional.temperature_differential(self.weighted_densities)
        f_T[:] += dfdT

        return f_T

    def correlation(self, i):
        """
        Access to one particle correlation
        """
        return self.comp_differentials[i].corr

    def set_up_special_weights(self, conv_type):
        """
        Args:
           conv_type (ConvType): Type of convolution required
        """
        special_weights_generated = False
        # Loop all weight functions
        for wf in self.comp_wfs[0]:
            if self.comp_wfs[0][wf].convolve:
                special_weights_generated = self.comp_wfs[0][wf].fourier_weights_has_been_calculated(conv_type)
                break

        if not special_weights_generated:
            for i in range(self.functional.nc):
                # Loop all weight functions
                for wf in self.comp_wfs[i]:
                    self.comp_wfs[i][wf].generate_special_fourier_weights(conv_type)

    def convolve_densities_by_type(self, rho, conv_type):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """
        self.set_up_special_weights(conv_type)
        self.weighted_densities.reset(rho)
        for i in range(self.functional.thermo.nc):
            self.weighted_densities.convolve_densities_by_type(rho.densities[i], i, conv_type)

        self.weighted_densities.update_utility_variables()

        # Calculate differentials
        # self.update_functional_differentials()
        # for i in range(self.functional.nc):
        #     # Loop all weight functions
        #     for wf in self.comp_wfs[i]:
        #         if self.comp_wfs[i][wf].convolve:
        #             self.comp_wfs[i][wf].convolve_differentials(self.comp_differentials[i].d_effective(wf),
        #                                                         self.comp_differentials[i].d_conv[wf])
            # Calculate one-particle correlation
            #self.comp_differentials[i].update_after_convolution()

    def plot_weighted_densities(self):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """
        plt.figure()
        # Loop all weight functions
        for wf in self.comp_wfs[0]:
            alias = self.comp_wfs[0].wfs[wf].alias
            if alias in self.comp_wfs[0].fmt_aliases:
                plt.plot(self.grid.z,self.weighted_densities.n[alias],label=alias)
            else:
                plt.plot(self.grid.z,self.weighted_densities.n[alias][0,:],label=alias)
        leg = plt.legend(loc="best", numpoints=1, frameon=False)
        plt.show()

    def plot_differentials(self):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """
        plt.figure()
        f0 = np.zeros(self.grid.n_grid)
        for i in range(self.functional.nc):
            for wf in self.comp_wfs[i].wfs:
                alias = self.comp_wfs[i].wfs[wf].alias
                plt.plot(self.grid.z,self.comp_differentials[i].d[alias][:],label=alias)
            plt.plot(self.grid.z,self.comp_differentials[i].mu_of_rho[:],label="rho")
        leg = plt.legend(loc="best", numpoints=1, frameon=False)
        plt.show()

if __name__ == "__main__":
    pass
