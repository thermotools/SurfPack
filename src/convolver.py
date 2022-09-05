#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import Geometry
from fmt_functionals import WeightFunctions
from grid import Grid
import numpy as np
from scipy.fft import dct, idct, dst, idst, fft, ifft


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
        # Utilities
        # self.n3neg = np.zeros(n_grid)
        # self.n3neg2 = np.zeros(n_grid)
        # self.n2v2 = np.zeros(n_grid)
        # self.logn3neg = np.zeros(n_grid)
        # self.n32 = np.zeros(n_grid)
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

    # def update_utility_variables(self):
    #     """
    #     """
    #     self.n3neg[:] = 1.0 - self.n3[:]
    #     self.n3neg2[:] = self.n3neg[:] ** 2
    #     self.n2v2[:] = self.n2v[:] ** 2
    #     self.logn3neg[:] = np.log(self.n3neg[:])
    #     self.n32[:] = self.n3[:] ** 2

    def update_after_convolution(self):
        """
        """
        # Account for segments
        for wd in self.n:
            if wd in self.wfs.fmt_aliases:
                self.n[wd][:] *= self.ms

        # self.n2[:] *= self.ms
        # self.n3[:] *= self.ms
        # self.n2v[:] *= self.ms
        # Get remaining densities from convoultion results
        # self.n1v[:] = self.n2v[:] / (4 * np.pi * self.R)
        # self.n0[:] = self.n2[:] / (4 * np.pi * self.R ** 2)
        # self.n1[:] = self.n2[:] / (4 * np.pi * self.R)
    #     self.update_utility_variables()

    def __getitem__(self, wd):
        return self.n[wd]

    def set_zero(self):
        """
        Set weights to zero
        """
        # self.n2[:] = 0.0
        # self.n3[:] = 0.0
        # self.n2v[:] = 0.0
        # self.n0[:] = 0.0
        # self.n1[:] = 0.0
        # self.n1v[:] = 0.0
        for wd in self.n:
            self.n[wd].fill(0.0)

    def __iadd__(self, other):
        """
        Add weights
        """
        # self.n2[:] += other.n2[:]
        # self.n3[:] += other.n3[:]
        # self.n2v[:] += other.n2v[:]
        # self.n0[:] += other.n0[:]
        # self.n1[:] += other.n1[:]
        # self.n1v[:] += other.n1v[:]
        # Add FMT densities
        for wd in self.n:
            if wd in self.wfs.fmt_aliases:
                self.n[wd][:] = other.n[wd][:]
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


        # if print_utilities:
        #     if index is None:
        #         print("n3neg: ", self.n3neg)
        #         print("n3neg2: ", self.n3neg2)
        #         print("n2v2: ", self.n2v2)
        #         print("logn3neg: ", self.logn3neg)
        #         print("n32: ", self.n32)
        #     else:
        #         print("n3neg: ", self.n3neg[index])
        #         print("n3neg2: ", self.n3neg2[index])
        #         print("n2v2: ", self.n2v2[index])
        #         print("logn3neg: ", self.logn3neg[index])
        #         print("n32: ", self.n32[index])

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

class WeightedDensitiesMaster(WeightedDensities):
    """
    """

    def __init__(self, n_grid: int, wfs: WeightFunctions, nc: int):
        """

        Args:
            n_grid:
            wfs:
            ms:
        """
        WeightedDensities.__init__(self, n_grid, wfs, ms=1.0)
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
                self.n[alias] = np.zeros((nc, n_grid))

    def update_utility_variables(self):
        """
        """
        self.n3neg[:] = 1.0 - self.n3[:]
        self.n3neg2[:] = self.n3neg[:] ** 2
        self.n2v2[:] = self.n2v[:] ** 2
        self.logn3neg[:] = np.log(self.n3neg[:])
        self.n32[:] = self.n3[:] ** 2

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
        # Utilities
        self.d2eff = np.zeros(n_grid)
        self.d2veff = np.zeros(n_grid)

        # Convolved properties
        # self.d3_conv = np.zeros(n_grid)
        # self.d2eff_conv = np.zeros(n_grid)
        # self.d2veff_conv = np.zeros(n_grid)
        # self.d_conv["w3"] = self.d3_conv
        # self.d_conv["w2eff"] = self.d2eff_conv
        # self.d_conv["wv2eff"] = self.d2veff_conv
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

    def set_functional_differentials(self, functional, ic):
        """

        Args:
            functional:
            ic:

        Returns:

        """
        self.d0[:] = functional.d0[:]
        self.d1[:] = functional.d1[:]
        self.d2[:] = functional.d2[:]
        self.d3[:] = functional.d3[:]
        self.d1v[:] = functional.d1v[:]
        self.d2v[:] = functional.d2v[:]
        self.d2eff[:] = functional.d2eff[:]
        self.d2veff[:] = functional.d2veff[:]

        for wf in self.wfs:
            alias = self.wfs[wf].alias
            if alias not in self.wfs.fmt_aliases:
                self.d[alias][:] = functional.diff[alias][:, ic]


    def print(self, index=None):
        """
        """
        print("\nDifferentials")
        for wd in self.d:
            if index is None:
                print(wd.replace("w", "d") +": ", self.d[wd][:])
            else:
                print(wd.replace("w", "d") +": ", self.d[wd][index])
        print("d2_eff: ", self.d2eff)
        print("d2v_eff: ", self.d2veff)
        # print("\nConvolution results")
        # print("d2eff_conv: ", self.d2eff_conv)
        # print("d3_conv: ", self.d3_conv)
        # print("d2veff_conv: ", self.d2veff_conv)
        # print("corr: ", self.corr)

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
        self.comp_weighted_densities = []
        self.comp_differentials = []
        self.comp_wfs = []
        for i in range(functional.thermo.nc):
            self.comp_wfs.append(WeightFunctions.Copy(functional.wf))
            # Set up Fourier weights
            for wfun in self.comp_wfs[i]:
                self.comp_wfs[i][wfun].generate_fourier_weights(grid, R[i], R_T[i])
            self.comp_weighted_densities.append(WeightedDensities(n_grid=grid.n_grid, wfs=self.comp_wfs[i], ms=functional.thermo.m[i]))
            self.comp_differentials.append(CompWeightedDifferentials(n_grid=grid.n_grid, wfs=self.comp_wfs[i], ms=functional.thermo.m[i]))


        # Overall weighted densities
        self.weighted_densities = WeightedDensitiesMaster(n_grid=grid.n_grid, wfs=functional.wf, nc=functional.thermo.nc)
        self.weighted_densities_T = WeightedDensitiesMaster(n_grid=grid.n_grid, wfs=functional.wf, nc=functional.thermo.nc)

    def convolve_density_profile(self, rho):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """
        self.weighted_densities.set_zero()
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
            # Loop all weight functions
            for wf in self.comp_wfs[i]:
                self.comp_wfs[i].wfs[wf].convolve_densities(rho_inf,
                                                            frho_delta,
                                                            self.comp_weighted_densities[i].n[wf])
            for wf in self.functional.wf.wfs:
                self.comp_wfs[i].wfs[wf].update_dependencies(self.comp_weighted_densities[i])
            # Account for segments
            self.comp_weighted_densities[i].update_after_convolution()
            # print("n0",self.comp_weighted_densities[i].n["w0"])
            # print("n1",self.comp_weighted_densities[i].n["w1"])
            # print("n2",self.comp_weighted_densities[i].n["w2"])
            # print("n3",self.comp_weighted_densities[i].n["w3"])
            # print("nv1",self.comp_weighted_densities[i].n["wv1"])
            # print("nv2",self.comp_weighted_densities[i].n["wv2"])
            # print("n_disp",self.comp_weighted_densities[i].n["w_disp"])

            # Add component contribution to overall density
            self.weighted_densities += self.comp_weighted_densities[i]
            for wf in self.comp_wfs[i]:
                alias = self.comp_wfs[i].wfs[wf].alias
                if alias not in self.comp_wfs[i].fmt_aliases:
                    self.weighted_densities.n[alias][i, :] = \
                        self.comp_weighted_densities[i].n[alias][:]
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

        # print("d2eff",self.comp_differentials[0].d_conv["w2"])
        # print("d3",self.comp_differentials[0].d_conv["w3"])
        # print("d2veff",self.comp_differentials[0].d_conv["wv2"])
        # print("d_disp",self.comp_differentials[0].d_conv["w_disp"])

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

    def convolve_density_profile_T(self, rho):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """

        self.weighted_densities_T.set_zero()
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
            # Loop all weight functions
            for wf in self.comp_wfs[i]:
                self.comp_wfs[i].wfs[wf].convolve_densities_T(rho_inf,
                                                              frho_delta,
                                                              self.comp_weighted_densities[i].n[wf])
            # for wf in self.functional.wf.wfs:
            #     self.comp_wfs[i].wfs[wf].update_dependencies(self.comp_weighted_densities[i])
            # Account for segments
            self.comp_weighted_densities[i].update_after_convolution()
            # print("n0",self.comp_weighted_densities[i].n["w0"])
            # print("n1",self.comp_weighted_densities[i].n["w1"])
            # print("n2",self.comp_weighted_densities[i].n["w2"])
            # print("n3",self.comp_weighted_densities[i].n["w3"])
            # print("nv1",self.comp_weighted_densities[i].n["wv1"])
            # print("nv2",self.comp_weighted_densities[i].n["wv2"])
            # print("n_disp",self.comp_weighted_densities[i].n["w_disp"])

            # Add component contribution to overall density
            self.weighted_densities_T += self.comp_weighted_densities[i]
            for wf in self.comp_wfs[i]:
                alias = self.comp_wfs[i].wfs[wf].alias
                if alias not in self.comp_wfs[i].fmt_aliases:
                    self.weighted_densities_T.n[alias][i, :] = \
                        self.comp_weighted_densities[i].n[alias][:]

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

if __name__ == "__main__":
    pass
