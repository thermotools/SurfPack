#!/usr/bin/env python3
import numpy as np
from cdft import cdft1D
import matplotlib.pyplot as plt
from utility import boundary_condition, density_from_packing_fraction, \
    get_data_container, plot_data_container, allocate_real_convolution_variable
from constants import DEBUG
import sys


class picard_geometry_solver():
    """
    Base solver class for minimisation objects.
    """

    def __init__(self, cDFT: cdft1D, new_fraction=0.1):
        """
        Initialises arrays and fourier objects required for minimisation procedure.

        Args:
            cDFT (cDFT1D object):    DFT object to be minimised

        Returns:
            None
        """
        self.cDFT = cDFT

        # Mask for updating density on inner domain
        self.NiWall = self.cDFT.N - self.cDFT.end
        self.domain_mask = np.full(self.cDFT.N, False, dtype=bool)
        self.domain_mask[self.NiWall:self.cDFT.end] = True
        self.wall_mask = np.full(self.cDFT.N, False, dtype=bool)
        if self.cDFT.left_boundary == boundary_condition["WALL"]:
            self.wall_mask[self.NiWall] = True
        if self.cDFT.right_boundary == boundary_condition["WALL"]:
            self.wall_mask[self.cDFT.end - 1] = True
        self.left_boundary_mask = np.full(self.cDFT.N, False, dtype=bool)
        self.left_boundary_mask[:self.NiWall] = True
        self.right_boundary_mask = np.full(self.cDFT.N, False, dtype=bool)
        self.right_boundary_mask[self.cDFT.N - self.NiWall:] = True
        # Set radius for plotting
        self.r = np.linspace(-self.NiWall * self.cDFT.dr, (self.cDFT.end - 1) * self.cDFT.dr, self.cDFT.N)

        # Density profile
        self.density = np.zeros(self.cDFT.N)
        self.density[:] = self.cDFT.bulk_density
        if self.cDFT.left_boundary == boundary_condition["WALL"]:
            self.density[self.left_boundary_mask] = 0.0
        if self.cDFT.right_boundary == boundary_condition["WALL"]:
            self.density[self.right_boundary_mask] = 0.0
        self.old_density = np.zeros(self.cDFT.N)
        self.old_density[:] = self.density[:]
        self.new_density = np.zeros(self.cDFT.N)
        self.mod_density = allocate_real_convolution_variable(self.cDFT.N)

        # Set up FFT objects if required
        self.cDFT.weights.setup_fft(self.cDFT.weighted_densities,
                                    self.cDFT.differentials,
                                    self.mod_density)
        # Configure PyFFTW to use multiple threads
        # fftw.config.NUM_THREADS = 2

        # Set state of solver
        self.converged = False

        # Set Picard parameter for relaxed successive substitution
        self.new_fraction = new_fraction

        # Error norm (np.inf: Max norm, None: 2 norm, ....)
        self.norm = np.inf

        self.generate_case_name()

    def picard_update(self):
        """
        Updates density profile using Picard procedures.

        Returns:
            error (float): Deviation between current and previous density profile
            have_failed (bool): Fail indicator
        """

        if DEBUG: self.cDFT.weights.print()
        # Calculate weighted densities
        self.mod_density[:] = self.density[:]
        self.mod_density[self.wall_mask] *= 0.5  # todo find explanation for 0.5 trick
        self.cDFT.weights.convolutions(self.cDFT.weighted_densities, self.mod_density)
        if DEBUG: self.cDFT.weighted_densities.print()

        # Calculate one-body direct correlation function
        self.cDFT.functional.differentials(self.cDFT.weighted_densities, self.cDFT.differentials)
        self.cDFT.weights.correlation_convolution(self.cDFT.differentials)
        if DEBUG: self.cDFT.differentials.print()

        # Calculate new density profile using the variations of the functional
        self.new_density[self.domain_mask] = self.cDFT.bulk_density * \
                                             np.exp(self.cDFT.differentials.corr[self.domain_mask]
                                                    + self.cDFT.beta *
                                                    (self.cDFT.excess_mu - self.cDFT.Vext[self.domain_mask]))
        if DEBUG: print("new_density", self.new_density)

        # Set old density profile
        self.old_density[:] = self.density[:]

        # Check for valid floats
        if np.any(np.isnan(self.new_density)) or np.any(np.isinf(self.new_density)):
            have_failed = True
        else:
            have_failed = False

        # Picard update
        self.density[self.domain_mask] = (1.0 - self.new_fraction) * self.density[self.domain_mask] + \
                                         self.new_fraction * self.new_density[self.domain_mask]

        # Calculate deviation between new and old density profiles
        error = np.linalg.norm(self.density - self.old_density, ord=self.norm)
        return error, have_failed

    def iteration_plot_profile(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("$z$")
        ax.set_ylabel(r"$\rho^*/\rho_{\rm{b}}^*$")
        d_plot, = ax.plot(self.r[self.domain_mask],
                          self.density[self.domain_mask] / self.cDFT.bulk_density,
                          lw=2, color="k")
        d_plot_old, = ax.plot(self.r[self.domain_mask],
                              self.old_density[self.domain_mask] / self.cDFT.bulk_density,
                              lw=2, color="k", ls="--")
        plt.show()

    def minimise(self, tolerance=1e-12,
                 maximum_iterations=10000000,
                 print_frequency=1000,
                 plot_profile=False):
        """
        Method to calculate the equilibrium density profile.

        Args:
            tolerance (float): Solver tolerance
            maximum_iterations
            print_frequency
        Returns:
            None
        """

        if plot_profile:
            self.iteration_plot_profile()
        error = 1.0
        for iteration in range(maximum_iterations):
            error, have_failed = self.picard_update()
            if have_failed:
                print("Solver got invalid number and failed")
                break
            if error < tolerance:
                self.converged = True
                if plot_profile:
                    self.iteration_plot_profile()
                break

            if iteration % print_frequency == 0:
                print("{} complete. Deviation: {}\n".format(iteration, error))
                if plot_profile:
                    self.iteration_plot_profile()

        if not self.converged:
            print("Solver did not converge. Deviation at exit {}\n".format(error))

    def print_perform_minimization_message(self):
        """

        """
        print('A successful minimisation have not been yet converged, and the equilibrium profile is missing.')
        print('Please perform a minimisation before performing result operations.')

    def generate_case_name(self):
        """
        Generate case name from specifications
        """
        self.case_name = f'Planar_{self.cDFT.wall}_{"{:.3f}".format(self.cDFT.eta)}'
        self.case_name += '_' + self.cDFT.functional.short_name

    def save_equilibrium_density_profile(self):
        """
        Save equilibrium density profile to file
        """
        if not self.converged:
            self.print_perform_minimization_message()
            return

        filename = self.case_name + '.dat'
        np.savetxt(filename,
                   np.c_[self.r[self.domain_mask],
                         self.density[self.domain_mask],
                         self.density[self.domain_mask] / self.cDFT.bulk_density],
                   header="# r, rho, rho/rho_bulk")

    def plot_equilibrium_density_profile(self, data_dict=None):
        """
        Plot equilibrium density profile
        Args:
            data_dict: Additional data to plot
        """
        if not self.converged:
            self.print_perform_minimization_message()
            return

        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("$z$")
        ax.set_ylabel(r"$\rho^*/\rho_{\rm{b}}^*$")
        ax.plot(self.r[self.domain_mask],
                self.density[self.domain_mask] / self.cDFT.bulk_density,
                lw=2, color="k", label="cDFT")
        if data_dict is not None:
            plot_data_container(data_dict, ax)
            leg = plt.legend(loc="best", numpoints=1)
            leg.get_frame().set_linewidth(0.0)

        filename = self.case_name + ".pdf"
        plt.savefig(filename)
        plt.show()

    def print_solver_status(self):

        """
        Print solver state and status

        Args:
            None
        Returns:
            None
        """

    def equilibrium_profile(self):
        """
        """

    def adsorption_sum_rule(self, ):
        """Prints results of adsorption sum rule
        """

    def pressure(self, ):
        """
        Returns the pressure of the system.
        """
        return self.DFT.pressure

    def chemical_potential(self, contributions="IE"):
        """
        Returns the chemical potential of the system.

        Args:
            Optional:
                contributions (int): I=Ideal and E=Excess, default IE

        Returns:
            chemical potential (float)
        """

        # if excess and ideal:
        #     return self.DFT.mu, self.DFT.T*np.log(self.DFT.bulk_density)
        # elif excess:
        #     return self.DFT.mu
        # elif ideal:
        #     return self.DFT.T*np.log(self.DFT.bulk_density)
        # else:
        #     return (self.DFT.mu + self.DFT.T*np.log(self.DFT.bulk_density))

    def surface_tension(self):
        """
        Returns the surface tension of the system.

        Args:
            Optional:
                write_to_file(bool): If True, result is written to
                    the output file. Default is False.

        Returns:
            surface_tension (float)
        """

    def grand_potential(self):
        """
        Returns the grand potential of the system.

        Args:
            write_to_file(bool): If True, result is written to the output file. Default is False.

        Returns:
            grand_potential (float)
        """

    def adsorption(self, ):
        """
        Returns the adsorption of the system.

        Args:
            None

        Returns:
            adsorption (float)
        """

    def contact_density(self, ):
        """
        Returns the contact density of the system.

        Args:
            None

        Returns:
            contact_density (float)
        """


if __name__ == "__main__":
    bulk_density = density_from_packing_fraction(eta=0.2)
    dft = cdft1D(bulk_density=bulk_density, functional="RF", domain_length=50.0, wall="HardWall", grid_dr=0.001)
    # dft = cdft1D(bulk_density=bulk_density, domain_length=50.0, wall="SlitHardWall", grid_dr=0.001)
    # dft = cdft1D(bulk_density=bulk_density, domain_length=1.0, wall="HardWall", grid_dr=0.5)
    solver = picard_geometry_solver(cDFT=dft)
    solver.minimise(print_frequency=40,
                    plot_profile=True)

    data_dict = get_data_container("../testRF.dat", labels=["RF"], x_index=0, y_indices=[2])
    solver.plot_equilibrium_density_profile(data_dict=data_dict)
