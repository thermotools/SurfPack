#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import pyfftw as fftw
from pyctp import pcsaft, saftvrmie, saftvrqmie

boundary_condition = {"OPEN": 0,
                      "WALL": 1}

def density_from_packing_fraction(eta, d=np.array([1.0])):
    """
    Calculates the reduced density of hard-sphere fluid from the packing fraction.

    Args:
        eta (ndarray): Hard-sphere packing fraction
        d (ndarray): Reduced hard-sphere diameter
    Returns:
        density(ndarray): reduced density
    """

    dens = np.zeros_like(eta)
    dens[:] = 6 * eta[:] / (np.pi * d[:] ** 3)

    return dens


def packing_fraction_from_density(dens, d=np.array([1.0])):
    """
    Calculates the hard-sphere fluid packing fraction from reduced density.

    Args:
        dens(ndarray): reduced density
        d (ndarray): Reduced hard-sphere diameter
    Returns:
        eta (ndarray): Hard-sphere packing fraction
    """

    eta = np.zeros_like(dens)
    eta[:] = dens[:] * (np.pi * d[:] ** 3) / 6

    return eta


def get_data_container(filename, labels=None, x_index=0, y_indices=None, colors=None):
    """
    ¨
    Args:
        filename (str):  Name of file
        labels (list of str): Label of data
        x_index (int): x index to plot against
        y_indices (list of int): Array of indices to plot
        colors (list of str): Color list
    Returns:
        data (dict): Container with information of data to plot
    """
    data = {}
    data["filename"] = filename
    data["labels"] = labels
    data["x"] = x_index
    if y_indices is None:
        data["y"] = [1]
    else:
        data["y"] = y_indices
    if colors is None:
        data["colors"] = ["r", "g", "b"]
    else:
        data["colors"] = colors
    return data


def plot_data_container(data_dict, ax):
    """

    Args:
        data_dict (dict):
        ax (plt.axis): Matplotlib axis

    Returns:

    """
    data = load_file(data_dict["filename"])
    label = None
    for yi, y in enumerate(data_dict["y"]):
        if data_dict["labels"] is not None:
            label = data_dict["labels"][yi]
        ax.plot(data[:, data_dict["x"]], data[:, y], lw=2,
                color=data_dict["colors"][yi], label=label)


def plot_data_container_list(data_dict_list, ylabel, xlabel="$z$", filename=None):
    """

    Args:
        data_dict_list (list of dict): List of data-dicts to plot
        ylabel (str): y-label 
        xlabel (str): x-label
        filename (str): File name 

    Returns:

    """
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(xlabel)
    # ax.set_ylabel(r"$\rho^*/\rho_{\rm{b}}^*$")
    ax.set_ylabel(ylabel)

    for data_dict in data_dict_list:
        plot_data_container(data_dict, ax)

    leg = plt.legend(loc="best", numpoints=1)
    leg.get_frame().set_linewidth(0.0)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def load_file(filename):
    """

    Args:
        filename (str): File to be read

    Returns:
        data (np.ndarray): Data read from file
    """
    file = open(filename, 'r')
    # Parse header
    n_lines = 0
    for line in file:
        words = line.split()
        if words[0][0] == '#':
            n_lines += 1
        else:
            break
    data = np.loadtxt(filename, skiprows=n_lines)
    return data


class densities():
    """
    Utility class. List of np.ndarrays.
    """

    def __init__(self, nc, N, is_conv_var=False):
        """

        Args:
            nc (int): Number of components
            N (int): Number of grid points
            is_conv_var (bool): Allocate using specific method for convolution variables
        """
        self.nc = nc
        self.N = N
        self.densities = []
        for i in range(nc):
            self.densities.append(np.zeros(N))

    def __setitem__(self, ic, rho):
        self.densities[ic][:] = rho[:]

    def __getitem__(self, ic):
        return self.densities[ic]

    def assign_elements(self, other, mask=None):
        """
        Assign all arrays ot other instance of densities. Apply mask if given.
        Args:
            other (densities): Other densities
            mask (bool ndarray): Array values to be changes

        """
        for i in range(self.nc):
            if mask is None:
                self.densities[i][:] = other.densities[i][:]
            else:
                self.densities[i][mask] = other.densities[i][mask]

    def assign_components(self, rho):
        """
        Assign all arrays ot other instance of densities. Apply mask if given.
        Args:
            rho (ndarray): Other densities

        """
        for i in range(self.nc):
            self.densities[i][:] = rho[i]

    def set_mask(self, mask, value=0.0):
        """
        Assign all arrays with value. Apply mask.
        Args:
            mask (bool ndarray): Array values to be changes
            value (float): Set masked array to value

        """
        for i in range(self.nc):
            self.densities[i][mask[i]] = value

    def mult_mask(self, mask, value):
        """
        Multiply arrays with value. Apply mask.
        Args:
            mask (bool ndarray): Array values to be changes
            value (float): Multiply masked array with value

        """
        for i in range(self.nc):
            self.densities[i][mask] *= value

    def is_valid_reals(self):
        """

        Returns:
            bool: True if no elements are NaN of Inf
        """
        is_valid = True
        for i in range(self.nc):
            if np.any(np.isnan(self.densities[i])) or np.any(np.isinf(self.densities[i])):
                is_valid = False
                break
        return is_valid

    def diff_norms(self, other, order=np.inf):
        """

        Args:
            other (densities): Other densities
            order: Order of norm

        Returns:

        """
        nms = np.zeros(self.nc)
        for i in range(self.nc):
            nms[i] = np.linalg.norm(
                self.densities[i] - other.densities[i], ord=order)
        return nms

    def diff_norm_scaled(self, other, scale, mask, ord=np.inf):
        """

        Args:
            other (densities): Other densities
            scale (ndarray): Column scale
            ord: Order of norm

        Returns:

        """
        nd_copy = self.get_nd_copy()[:, mask]
        nd_copy_other = other.get_nd_copy()[:, mask]
        nd_copy -= nd_copy_other
        nd_copy = (nd_copy.T / scale).T
        norm = np.linalg.norm(nd_copy, ord=ord)
        return norm

    def get_nd_copy(self):
        """
        Make ndarray with component densities as columns
        Returns:
            ndarray: nc x N array.
        """
        nd_copy = np.zeros((self.nc, self.N))
        for i in range(self.nc):
            nd_copy[i, :] = self.densities[i][:]
        return nd_copy


class weighted_densities_1D():
    """
    """

    def __init__(self, N, R, ms=1.0, mask_conv_results=None):
        """

        Args:
            N:
            R:
            ms:
            mask_conv_results:
        """
        self.N = N
        self.R = R
        self.ms = ms
        self.n0 = np.zeros(N)
        self.n1 = np.zeros(N)
        self.n2 = np.zeros(N)
        self.n3 = np.zeros(N)
        self.n1v = np.zeros(N)
        self.n2v = np.zeros(N)
        # Utilities
        self.n3neg = np.zeros(N)
        self.n3neg2 = np.zeros(N)
        self.n2v2 = np.zeros(N)
        self.logn3neg = np.zeros(N)
        self.n32 = np.zeros(N)
        # Fourier space weighted density
        self.fn2 = np.zeros(N, dtype=np.cdouble)
        self.fn3 = np.zeros(N, dtype=np.cdouble)
        self.fn2v = np.zeros(N, dtype=np.cdouble)
        # Mask results from convolution
        if mask_conv_results is None:
            self.mask_conv_results = np.full(N, False, dtype=bool)
        else:
            self.mask_conv_results = mask_conv_results

        # Utility for testing
        self.n_max_test = 6

        # Preallocated delta-variables
        self.fn2_delta = np.zeros(N, dtype=np.cdouble)
        self.fn3_delta = np.zeros(N, dtype=np.cdouble)
        self.fn2v_delta = np.zeros(N, dtype=np.cdouble)
        self.fn2_delta_cs = np.zeros(N)
        self.fn3_delta_cs = np.zeros(N)
        self.fn2v_delta_cs = np.zeros(N)
        self.fn2_delta_sph = np.zeros(N)
        self.fn3_delta_sph = np.zeros(N)
        self.fn2v_delta_sph = np.zeros(N)
        
    def set_convolution_result_mask(self, mask_conv_results):
        """

        Args:
            mask_conv_results: Mask for setting zeros in output from convolution
        """
        self.mask_conv_results[:] = mask_conv_results[:]

    def update_utility_variables(self):
        """
        """
        self.n3neg[:] = 1.0 - self.n3[:]
        self.n3neg2[:] = self.n3neg[:] ** 2
        self.n2v2[:] = self.n2v[:] ** 2
        self.logn3neg[:] = np.log(self.n3neg[:])
        self.n32[:] = self.n3[:] ** 2

    def update_after_convolution(self):
        """
        """
        self.n2[self.mask_conv_results] = 0.0
        self.n3[self.mask_conv_results] = 0.0
        self.n2v[self.mask_conv_results] = 0.0
        # Account for segments
        self.n2[:] *= self.ms
        self.n3[:] *= self.ms
        self.n2v[:] *= self.ms
        # Get remaining densities from convoultion results
        self.n1v[:] = self.n2v[:] / (4 * np.pi * self.R)
        self.n0[:] = self.n2[:] / (4 * np.pi * self.R ** 2)
        self.n1[:] = self.n2[:] / (4 * np.pi * self.R)
        self.update_utility_variables()

    def set_zero(self):
        """
        Set weights to zero
        """
        self.n2[:] = 0.0
        self.n3[:] = 0.0
        self.n2v[:] = 0.0
        self.n0[:] = 0.0
        self.n1[:] = 0.0
        self.n1v[:] = 0.0

    def __iadd__(self, other):
        """
        Add weights
        """
        self.n2[:] += other.n2[:]
        self.n3[:] += other.n3[:]
        self.n2v[:] += other.n2v[:]
        self.n0[:] += other.n0[:]
        self.n1[:] += other.n1[:]
        self.n1v[:] += other.n1v[:]
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
        self.R = 0.5
        self.update_utility_variables()

    def get_density(self, i):
        """
        Get weighted density number i
        """
        if i == 0:
            n = self.n0
        elif i == 1:
            n = self.n1
        elif i == 2:
            n = self.n2
        elif i == 3:
            n = self.n3
        elif i == 4:
            n = self.n1v
        elif i == 5:
            n = self.n2v
        else:
            raise ValueError("get_density: Index out of bounds")
        return n

    def set_density(self, i, n):
        """
        Set weighted density number i
        """
        if i == 0:
            self.n0[:] = n[:]
        elif i == 1:
            self.n1[:] = n[:]
        elif i == 2:
            self.n2[:] = n[:]
        elif i == 3:
            self.n3[:] = n[:]
        elif i == 4:
            self.n1v[:] = n[:]
        elif i == 5:
            self.n2v[:] = n[:]

    def print(self, print_utilities=False, index=None):
        """

        Args:
            print_utilities (bool): Print also utility variables
        """
        print("\nWeighted densities:")
        if index is None:
            print("n0: ", self.n0)
            print("n1: ", self.n1)
            print("n2: ", self.n2)
            print("n3: ", self.n3)
            print("n1v: ", self.n1v)
            print("n2v: ", self.n2v)
        else:
            print("n0: ", self.n0[index])
            print("n1: ", self.n1[index])
            print("n2: ", self.n2[index])
            print("n3: ", self.n3[index])
            print("n1v: ", self.n1v[index])
            print("n2v: ", self.n2v[index])

        if print_utilities:
            if index is None:
                print("n3neg: ", self.n3neg)
                print("n3neg2: ", self.n3neg2)
                print("n2v2: ", self.n2v2)
                print("logn3neg: ", self.logn3neg)
                print("n32: ", self.n32)
            else:
                print("n3neg: ", self.n3neg[index])
                print("n3neg2: ", self.n3neg2[index])
                print("n2v2: ", self.n2v2[index])
                print("logn3neg: ", self.logn3neg[index])
                print("n32: ", self.n32[index])

    def plot(self, r, show=True, mask=None):
        """

        Args:
            r (np.ndarray): Spatial resolution
            show (bool): Execute plt.show ?
            mask (ndarray of bool): What to plot?
        """
        if mask is None:
            mask = np.full(len(r), True, dtype=bool)
        fig, ax1 = plt.subplots()
        ax1.plot(r[mask], self.n0[mask], label="n0", color="r")
        ax1.plot(r[mask], self.n1[mask], label="n1", color="g")
        ax1.plot(r[mask], self.n2[mask], label="n2", color="b")
        ax1.plot(r[mask], self.n3[mask], label="n3", color="orange")
        ax2 = ax1.twinx()
        ax2.plot(r[mask], self.n2v[mask], label="n2v", color="grey")
        ax2.plot(r[mask], self.n1v[mask], label="n1v", color="cyan")
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


class weighted_densities_pc_saft_1D(weighted_densities_1D):
    """
    """

    def __init__(self, N, R, ms, mask_conv_results=None):
        """
        """
        weighted_densities_1D.__init__(self, N, R, ms, mask_conv_results)
        self.rho_disp = np.zeros(N)
        self.rho_disp_array = None
        self.n_max_test = 7

    def set_zero(self):
        """
        Set weights to zero
        """
        weighted_densities_1D.set_zero(self)
        self.rho_disp[:] = 0.0
        if self.rho_disp_array is not None:
            self.rho_disp_array[:] = 0.0

    def __iadd__(self, other):
        """
        Add weights
        """
        weighted_densities_1D.__iadd__(self, other)
        self.rho_disp[:] += other.rho_disp[:]
        return self

    def set_testing_values(self, rho=None):
        """
        Set some dummy values for testing differentials
        """
        weighted_densities_1D.set_testing_values(self, rho)
        if self.rho_disp_array is None:
            self.rho_disp_array = np.ones((1, 1))
        if rho is not None:
            self.rho_disp[:] = rho
            self.rho_disp_array[:] = rho
        else:
            self.rho_disp[:] = 1.0

    def get_density(self, i):
        """
        Get weighted density number i
        """
        if i <= 5:
            n = weighted_densities_1D.get_density(self, i)
        else:
            n = self.rho_disp
        return n

    def set_density(self, i, n):
        """
        Set weighted density number i
        """
        if i <= 5:
            weighted_densities_1D.set_density(self, i, n)
        else:
            self.rho_disp[:] = n
            self.rho_disp_array[:] = n

    def print(self, print_utilities=False, index=None):
        """

        Args:
            print_utilities (bool): Print also utility variables
        """
        weighted_densities_1D.print(self, print_utilities, index)

        print("PC-SAFT:")
        if index is None:
            print("r_disp: ", self.rho_disp)
        else:
            print("r_disp: ", self.rho_disp[index])

    def plot(self, r, show=True, mask=None):
        """

        Args:
            r (np.ndarray): Spatial resolution
            show (bool): Execute plt.show ?
            mask (ndarray of bool): What to plot?
        """
        if mask is None:
            mask = np.full(len(r), True, dtype=bool)
        ax1, ax2 = weighted_densities_1D.plot(self, r, show=False, mask=mask)
        ax1.plot(r[mask], self.rho_disp[mask], label="rho_disp", color="k")
        if show:
            leg = ax1.legend(loc="center left", numpoints=1)
            leg.get_frame().set_linewidth(0.0)
            leg = ax2.legend(loc="center right", numpoints=1)
            leg.get_frame().set_linewidth(0.0)
            plt.show()
        return ax1, ax2


class differentials_1D():
    """
    """

    def __init__(self, N, R, mask_conv_results=None):
        """

        Args:
            N:
            R:
            mask_conv_results:
        """
        self.N = N
        self.R = R
        self.d0 = np.zeros(N)
        self.d1 = np.zeros(N)
        self.d2 = np.zeros(N)
        self.d3 = np.zeros(N)
        self.d1v = np.zeros(N)
        self.d2v = np.zeros(N)
        # Utilities
        self.d2eff = np.zeros(N)
        self.d2veff = np.zeros(N)
        # Utilities
        self.d3_conv = np.zeros(N)
        self.d2eff_conv = np.zeros(N)
        self.d2veff_conv = np.zeros(N)
       
        # One - body direct correlation function
        self.corr = np.zeros(self.N)
        # Fourier space differentials
        self.fd2eff = np.zeros(N, dtype=np.cdouble)
        self.fd3 = np.zeros(N, dtype=np.cdouble)
        self.fd2veff = np.zeros(N, dtype=np.cdouble)
        self.fd3_conv = np.zeros(N, dtype=np.cdouble)
        self.fd2eff_conv = np.zeros(N, dtype=np.cdouble)
        self.fd2veff_conv = np.zeros(N, dtype=np.cdouble)

        # Fourier space differentials (cosine-sine transform)
        self.fd2eff_cs = np.zeros(N)
        self.fd3_cs = np.zeros(N)
        self.fd2veff_cs = np.zeros(N)
        self.fd2veff_cs_V = np.zeros(N)
        self.fd3_conv_cs = np.zeros(N)
        self.fd2eff_conv_cs = np.zeros(N)
        self.fd2veff_conv_cs = np.zeros(N)

        # Fourier space differentials (spherical transform)
        self.fd2eff_sph = np.zeros(N)
        self.fd3_sph = np.zeros(N)
        self.fd2veff_sph = np.zeros(N)
        self.fd2veff_sph_V = np.zeros(N)
        self.fd3_conv_sph = np.zeros(N)
        self.fd2eff_conv_sph = np.zeros(N)
        self.fd2veff_conv_sph = np.zeros(N)
      
        # Mask results from convolution
        if mask_conv_results is None:
            self.mask_conv_results = np.full(N, False, dtype=bool)
        else:
            self.mask_conv_results = mask_conv_results

    def set_convolution_result_mask(self, mask_conv_results):
        """

        Args:
            mask_conv_results: Mask for setting zeros in output from convolution
        """
        self.mask_conv_results[:] = mask_conv_results[:]

    def update_after_convolution(self):
        """

        Returns:
        """

        self.d3_conv[self.mask_conv_results] = 0.0
        self.d2eff_conv[self.mask_conv_results] = 0.0
        self.d2veff_conv[self.mask_conv_results] = 0.0
        self.corr[:] = -(self.d3_conv[:] +
                         self.d2eff_conv[:] + self.d2veff_conv[:])
        # self.corr[:] = -(self.d3_conv[:] +
        #                  self.d2eff_conv[:])

    def set_functional_differentials(self, functional, ic=None):
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

    def get_differential(self, i):
        """
        Get differential number i
        """
        if i == 0:
            d = self.d0
        elif i == 1:
            d = self.d1
        elif i == 2:
            d = self.d2
        elif i == 3:
            d = self.d3
        elif i == 4:
            d = self.d1v
        elif i == 5:
            d = self.d2v
        else:
            raise ValueError("get_differential: Index out of bounds")
        return d

    def print(self):
        """
        """
        print("\nDifferentials")
        print("d0: ", self.d0)
        print("d1: ", self.d1)
        print("d2: ", self.d2)
        print("d3: ", self.d3)
        print("d1v: ", self.d1v)
        print("d2v: ", self.d2v)
        print("d2_eff: ", self.d2eff)
        print("d2v_eff: ", self.d2veff)
        print("\nConvolution results")
        print("d2eff_conv: ", self.d2eff_conv)
        print("d3_conv: ", self.d3_conv)
        print("d2veff_conv: ", self.d2veff_conv)
        print("corr: ", self.corr)

    def plot(self, r, show=True, mask=None):
        """

        Args:
            r (np.ndarray): Spatial resolution
            show (bool): Execute plt.show ?
            mask (ndarray of bool): What to plot?
        """
        if mask is None:
            mask = np.full(len(r), True, dtype=bool)
        fig, ax1 = plt.subplots()
        ax1.plot(r[mask], self.d2eff_conv[mask], label="d2_sum", color="r")
        ax1.plot(r[mask], self.d3_conv[mask], label="d3", color="g")
        #ax1.plot(r[mask], self.n2[mask], label="n2", color="b")
        #ax1.plot(r[mask], self.n3[mask], label="n3", color="orange")
        ax2 = ax1.twinx()
        ax2.plot(r[mask], self.d2veff_conv[mask],
                 label="d2v_sum", color="orange")
        #ax2.plot(r[mask], self.n1v[mask], label="n1v", color="cyan")
        ax1.set_xlabel("$r$")
        ax1.set_ylabel("$d$")
        ax2.set_ylabel("$d_v$")
        if show:
            ax1.plot(r[mask], self.corr[mask], label="corr", color="b")
            leg = ax1.legend(loc="best", numpoints=1)
            leg.get_frame().set_linewidth(0.0)
            leg = ax2.legend(loc="best", numpoints=1)
            leg.get_frame().set_linewidth(0.0)
            plt.show()
        return ax1, ax2


class quadratic_polynomial():
    """
    """

    def __init__(self, x, y):
        """

        Args:
            x: Abscissa values
            y: Ordinate values
        """
        a = np.zeros((len(x), 3))
        for i in range(len(x)):
            a[i, 0] = 1.0
            a[i, 1] = x[i]
            a[i, 2] = x[i]**2
        self.c = np.linalg.solve(a, y)

    def get_extrema(self):
        """

        Returns:
            (float): x at extrema
        """
        x_extrema = -self.c[1]/(2*self.c[2])
        return x_extrema

    def evaluate(self, x):
        """

        Args:
            x (np.ndarray): Abscissa values
        Returns:
            (np.ndarray): p(x of x)
        """
        p_of_x = np.zeros_like(x)
        p_of_x[:] = self.c[0] + self.c[1] * x[:] + self.c[2] * x[:]**2
        return p_of_x


class differentials_pc_saft_1D(differentials_1D):
    """
    """

    def __init__(self, N, R, mask_conv_results=None):
        """

        Args:
            N:
            R:
            mask_conv_results:
        """
        differentials_1D.__init__(self, N, R, mask_conv_results)
        self.mu_disp = np.zeros(N)
        self.mu_disp_conv = np.zeros(N)

    def set_functional_differentials(self, functional, ic=None):
        """

        Args:
            functional:
            ic:

        Returns:

        """
        differentials_1D.set_functional_differentials(self, functional, ic)
        self.mu_disp[:] = functional.mu_disp[:, ic]

    def update_after_convolution(self):
        """
        Add PC-SAFT contribution to one particle correlation function
        """
        differentials_1D.update_after_convolution(self)
        self.mu_disp_conv[self.mask_conv_results] = 0.0
        self.corr[:] -= self.mu_disp_conv

    def plot(self, r, show=True, mask=True):
        """

        Args:
            r (np.ndarray): Spatial resolution
            show (bool): Execute plt.show ?
            mask (ndarray of bool): What to plot?
        """
        if mask is None:
            mask = np.full(len(r), True, dtype=bool)
        ax1, ax2 = differentials_1D.plot(self, r, show=False, mask=mask)
        ax1.plot(r[mask], self.mu_disp_conv[mask], label="mu_disp", color="k")
        if show:
            ax1.plot(r[mask], self.corr[mask], label="corr", color="b")
            leg = ax1.legend(loc="center left", numpoints=1)
            leg.get_frame().set_linewidth(0.0)
            leg = ax2.legend(loc="center right", numpoints=1)
            leg.get_frame().set_linewidth(0.0)
            plt.show()
        return ax1, ax2


def get_thermopack_model(model):
    """
        Return thermopack object
    Args:
        model (str): Model name "PC-SAFT", "SAFT-VR Mie", ...

    Returns:

    """
    if model.upper() in ("PC-SAFT", "PCSAFT"):
        thermo = pcsaft.pcsaft()
    elif model.upper() in ("SAFTVRMIE", "SAFT-VR MIE", "SAFT-VR-MIE"):
        thermo = saftvrmie.saftvrmie()
    elif model.upper() in ("SAFTVRQMIE", "SAFT-VRQ MIE", "SAFT-VRQ-MIE"):
        thermo = saftvrqmie.saftvrqmie()
    else:
        raise ValueError("Unknown thermopack model: " + model)
    return thermo


def get_initial_densities_vle(z, rho_g, rho_l, R, reduced_temperature):
    """
    Calculate initial densities for gas-liquid interface calculatsion
    Args:
        z (np.ndarray): Grid positions (Symmetric around z=0)
        R (np.ndarray): Hard sphere radius in reduced units
        rho_g (list of float): Gas density
        rho_l (list of float): Liquid density
        reduced_temperature (float): T/Tc
    Returns:
       rho0 (densities): Initial density with liquid deinsity to the right

    """
    rho0 = densities(np.shape(R)[0], np.shape(z)[0])
    for i in range(np.shape(R)[0]):
        #         rho0.densities[i][:] = 0.5*(rho_g[i] + rho_l[i]) + 0.5 * \
        #             (rho_l[i] - rho_g[i])*np.tanh(0.3*z[:]/R[i])
        rho0.densities[i][:] = 0.5*(rho_g[i] + rho_l[i]) + 0.5 * \
            (rho_l[i] - rho_g[i])*np.tanh(z[:]/(2*R[i])
                                          * (2.4728 - 2.3625 * reduced_temperature))
    return rho0


if __name__ == "__main__":
    pass
