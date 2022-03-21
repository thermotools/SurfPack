#!/usr/bin/env python3
import numpy as np
from cdft import cdft1D
import matplotlib.pyplot as plt
from utility import boundary_condition, density_from_packing_fraction, \
    get_data_container, plot_data_container, allocate_real_convolution_variable, \
    quadratic_polynomial
from constants import DEBUG
import ng_extrapolation
from matplotlib.animation import FuncAnimation
import copy
import sys


class picard_geometry_solver():
    """
    Base solver class for minimisation objects.
    """

    def __init__(self, cDFT: cdft1D, alpha_min=0.1, alpha_max=0.9, ng_extrapolations=None, line_search="None"):
        """
        Initialises arrays and fourier objects required for minimisation procedure.

        Args:
            cDFT (cDFT1D object):    DFT object to be minimised
            alpha_min (float): Minimum value for Picard parameter
            alpha_max (float): Maximum value for Picard parameter
            ng_extrapolations (int): Extrapolation with method of Ng? Value set update frequency.
            line_search (str): Perform line search ("None", "GP", "Error")
        Returns:
            None
        """
        self.cDFT = cDFT

        # Mask for updating density on inner domain
        self.NiWall = self.cDFT.NiWall
        self.domain_mask = self.cDFT.domain_mask
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
        self.temp_density = np.zeros(self.cDFT.N)

        # Set up FFT objects if required
        self.cDFT.weights.setup_fft(self.cDFT.weighted_densities,
                                    self.cDFT.differentials)
        # Configure PyFFTW to use multiple threads
        # fftw.config.NUM_THREADS = 2

        # Set state of solver
        self.converged = False

        # Set state of solver
        self.iteration = 0
        self.error = 1.0

        # Set Picard parameter for relaxed successive substitution
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Error norm (np.inf: Max norm, None: 2 norm, ....)
        self.norm = np.inf

        # Set case name to be used for output files etc.
        self.generate_case_name()

        # Extrapolations according to Ng 1974
        self.ng = ng_extrapolation.ng_extrapolation(self.cDFT.N, ng_extrapolations, self.domain_mask)

        # Line search
        self.do_line_search = not line_search.upper() == "NONE"
        self.line_search_eval = line_search.upper()
        self.ls_density = np.zeros(self.cDFT.N)

        # Solver defaults
        self.tolerance = 1.0e-12
        self.print_frequency = 50
        self.maximum_iterations = 10000000

    def picard_density(self, mix_density, alpha, new_density=None, old_density=None):
        """
        Mix new_density and density
        Args:
            alpha (float): Fraction of new density
            mix_density (np.ndarray): Array to be mixed (output)
            new_density (np.ndarray): New density
            old_density (np.ndarray): Old density

        """
        if new_density is None:
            new_density = self.new_density
        if old_density is None:
            old_density = self.density

        mix_density[self.left_boundary_mask] = old_density[self.left_boundary_mask]
        mix_density[self.right_boundary_mask] = old_density[self.right_boundary_mask]
        mix_density[self.cDFT.domain_mask] = (1 - alpha) * old_density[self.cDFT.domain_mask] + \
                                             alpha * new_density[self.cDFT.domain_mask]


    def successive_substitution(self, density):
        """
        Updates density profile using Picard procedures.

        Returns:
            error (float): Deviation between current and previous density profile
            have_failed (bool): Fail indicator
        """

        if DEBUG: self.cDFT.weights.print()
        # Calculate weighted densities
        self.mod_density[:] = density[:]
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

    def picard_update(self):
        """
        Updates density profile using Picard procedures.

        Returns:
            error (float): Deviation between current and previous density profile
            have_failed (bool): Fail indicator
        """
        # Do successive substitution to get new_density
        self.successive_substitution(self.density)
        # Check for valid floats
        if np.any(np.isnan(self.new_density)) or np.any(np.isinf(self.new_density)):
            have_failed = True
            return have_failed
        else:
            have_failed = False

        # Set old density profile
        self.old_density[:] = self.density[:]

        # Update Ng history
        self.ng.push_back(self.density, self.new_density, self.iteration)

        if self.error < 1.0e-3 and self.ng.time_to_update(self.iteration):
            self.density[self.domain_mask] = self.ng.extrapolate()
        else:
            # Picard update
            # Do line search?
            if self.do_line_search:
                #self.plot_line_search()
                alpha = self.line_search(debug=False)
            else:
                alpha = self.alpha_min
            self.picard_density(self.density, alpha)

        # Calculate deviation between new and old density profiles
        self.error = np.linalg.norm(self.density - self.old_density, ord=self.norm)
        return have_failed

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
            maximum_iterations (int): Maximum number of iteration
            print_frequency (int): HOw often should solver status be printed?
            plot_profile (bool): Plot density profile while iterating?
        Returns:
            None
        """
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        if plot_profile:
            self.iteration_plot_profile()

        self.iteration = 0
        self.print_frequency = print_frequency
        for iteration in range(maximum_iterations):
            have_failed = self.single_iteration()
            if have_failed:
                print("Solver got invalid number and failed")
                break
            if self.converged:
                break
            if self.iteration % print_frequency == 0:
                if plot_profile:
                    self.iteration_plot_profile()

        if self.converged:
            if plot_profile:
                self.iteration_plot_profile()
            print(f"Solver converged after {self.iteration} iterations\n")
        else:
            print(f"Solver did not converge. Deviation at exit {self.error}\n")

    def single_iteration(self):
        """
        Method to calculate the equilibrium density profile.

        Args:
            tolerance (float): Solver tolerance
            maximum_iterations (int): Maximum number of iteration
            print_frequency (int): HOw often should solver status be printed?
            plot_profile (bool): Plot density profile while iterating?
        Returns:
            None
        """

        have_failed = self.picard_update()
        self.iteration += 1
        if self.error < self.tolerance:
            self.converged = True
        if self.iteration % self.print_frequency == 0:
            print(f"{self.iteration} complete. Deviation: {self.error}\n")
        return have_failed

    def animate(self,
                tolerance=1e-12,
                maximum_iterations=10000000,
                print_frequency=1000,
                z_max=None):
        """
        Method to calculate the equilibrium density profile.

        Args:
            tolerance (float): Solver tolerance
            maximum_iterations (int): Maximum number of iteration
            print_frequency (int): HOw often should solver status be printed?
            z_max (float): Limit plot range
        Returns:
            None
        """
        self.tolerance = tolerance
        self.iteration = 0
        self.print_frequency = print_frequency
        self.maximum_iterations = maximum_iterations
        anim = anim_solver(self, z_max=z_max)
        plt.show()


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


    # def test_differentials(self):
    #
    #     """
    #
    #
    #     Args:
    #         None
    #     Returns:
    #         None
    #     """
    #     # Calculate weighted densities
    #     density0 = copy.deepcopy(self.density)
    #     wd0 = copy.deepcopy(self.cDFT.weighted_densities)
    #     diff0 = copy.deepcopy(self.cDFT.differentials)
    #     self.cDFT.weights.convolutions(wd0, density0)
    #
    #     # Calculate one-body direct correlation function
    #     self.cDFT.functional.differentials(wd0, diff0)
    #     self.cDFT.weights.correlation_convolution(diff0)
    #
    #     density1 = copy.deepcopy(self.density)
    #     wd1 = copy.deepcopy(self.cDFT.weighted_densities)
    #     diff1 = copy.deepcopy(self.cDFT.differentials)
    #
    #     index = 10
    #     rho = density1[index]
    #     eps = 1.0e-5
    #
    #     density1[index] += rho*eps
    #     self.cDFT.weights.convolutions(wd1, density1)
    #
    #     # Calculate one-body direct correlation function
    #     self.cDFT.functional.differentials(wd1, diff1)
    #     self.cDFT.weights.correlation_convolution(diff1)

    def get_line_search_error(self, alpha, update=True):
        """
        Print solver state and status

        Args:
            None
        Returns:
            None
        """
        if update:
            self.picard_density(self.temp_density, alpha, new_density=self.ls_density)
        if self.line_search_eval == "ERROR":
            # Knepley et al. 2010. doi: 10.1063/1.3357981
            if update:
                self.successive_substitution(self.temp_density)
            err = np.linalg.norm(self.new_density[self.domain_mask]-self.temp_density[self.domain_mask], ord=2)**2
        elif self.line_search_eval.strip("-_") in ["GP", "GRANDPOTENTIAL"]:
            err = self.cDFT.grand_potential(self.temp_density, update_convolutions=update)[0]
        else:
            raise ValueError("get_line_search_error: Wrong type for function evaluation")
        return err

    def calc_alpha_max(self):

        """
        Calculate maximum alpha value such that n3 < 1.0
        See: Knepley et al. 2010. doi: 10.1063/1.3357981

        Returns:
            alpha_max (float): Maximum alpha value
        """
        n3_inf_0 = np.linalg.norm(self.cDFT.weighted_densities.n3, ord=np.inf)
        self.mod_density[:] = self.new_density[:]
        self.cDFT.weights.convolution_n3(self.cDFT.weighted_densities, self.mod_density)
        n3_inf_1 = np.linalg.norm(self.cDFT.weighted_densities.n3, ord=np.inf)
        alpha_max = min(0.9, (0.9 - n3_inf_0)/max(n3_inf_1 - n3_inf_0, 1e-6))
        #print("alpha_max", alpha_max, n3_inf_0,n3_inf_1)
        return alpha_max

    def line_search(self, debug=False):
        """
        Do quadratic line search and obtain best alpha

        Args:
            None
        Returns:
            None
        """
        self.ls_density[:] = self.new_density[:]
        self.temp_density[:] = self.density[:]  # Set to avoid update in get_line_search_error
        e0 = self.get_line_search_error(alpha=0.0, update=False)
        alpha_max = self.calc_alpha_max()
        alpha1 = 0.5*alpha_max
        e1 = self.get_line_search_error(alpha=alpha1, update=True)
        if e1 < e0:
            alpha2 = alpha_max
        else:
            alpha2 = 0.25 * alpha_max
        e2 = self.get_line_search_error(alpha=alpha2, update=True)
        errors = np.array([e0, e1, e2])
        alphas = np.array([0.0, alpha1, alpha2])
        if debug:
            plt.plot(alphas, errors, ls="None", marker="o")
            plt.show()
        if e2 > e0 and e1 > e0:
            alpha = self.alpha_min
        else:
            qp = quadratic_polynomial(alphas, errors)
            alpha_extreme = qp.get_extrema()
            if alpha_extreme > 1.0e-3:
                alpha = min(alpha_extreme, alpha_max)
            else:
                alpha = self.alpha_min
        # Reset new_density
        self.new_density[:] = self.ls_density[:]
        return alpha

    def plot_line_search(self):
        """
        Calculate function values and plot together with quadratic line search polynomial

        """
        alpha_max = self.calc_alpha_max()
        self.ls_density[:] = self.new_density[:]
        alphas = np.linspace(0.0, alpha_max, 3)
        objective = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            objective[i] = self.get_line_search_error(alpha=alpha, update=True)
        qp = quadratic_polynomial(alphas, objective)

        alphas = np.linspace(0.0, alpha_max, 100)
        objective = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            objective[i] = self.get_line_search_error(alpha=alpha, update=True)
        plt.plot(alphas, objective, color="b")
        plt.plot(alphas, qp.evaluate(alphas), label="Polynomial", color="g")
        print("extrema", qp.get_extrema())
        #plt.show()
        # Reset density
        self.new_density[:] = self.ls_density[:]


    def print_solver_status(self):

        """
        Print solver state and status

        Args:
            None
        Returns:
            None
        """

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


class anim_solver():
    def __init__(self, solver, z_max=None, **kw):
        self.solver = solver
        self.fig, self.ax = plt.subplots()
        self.line = None
        self.z_max = z_max
        self.ani = FuncAnimation(self.fig,
                                 self.animate,
                                 frames=solver.maximum_iterations,
                                 repeat=False,
                                 init_func=self.init_plot,
                                 interval=10,
                                 blit=False,
                                 **kw)

    def animate(self, frame):
        have_failed = self.solver.single_iteration()
        if have_failed or self.solver.converged:
            if have_failed: print(f"Simulation failed")
            if self.solver.converged: print(f"Simulation converged after {solver.iteration} iterations")
            self.ani.event_source.stop()
        self.line.set_data(self.solver.r[self.solver.domain_mask],
                           self.solver.density[self.solver.domain_mask] / self.solver.cDFT.bulk_density)
        #self.line, = self.ax.plot(self.solver.r[self.solver.domain_mask],
        #                          self.solver.density[self.solver.domain_mask] / self.solver.cDFT.bulk_density,
        #                          lw=2, color="k")
        self.ax.relim()
        self.ax.autoscale_view()
        if self.z_max is not None:
            self.ax.set_xlim(0, self.z_max)
        #self.ax.set_ylim(-1, 1)
        return self.line,

    def init_plot(self):
        self.line, = self.ax.plot(self.solver.r[self.solver.domain_mask],
                                  self.solver.density[self.solver.domain_mask] / self.solver.cDFT.bulk_density,
                                  lw=2, color="k")
        self.ax.set_xlabel("$z$")
        self.ax.set_ylabel(r"$\rho^*/\rho_{\rm{b}}^*$")
        return self.line,

if __name__ == "__main__":
    bulk_density = density_from_packing_fraction(eta=0.2)
    dft = cdft1D(bulk_density=bulk_density, functional="RF", domain_length=50.0, wall="HardWall", grid_dr=0.001)
    # dft = cdft1D(bulk_density=bulk_density, domain_length=50.0, wall="SlitHardWall", grid_dr=0.001)
    # dft = cdft1D(bulk_density=bulk_density, domain_length=1.0, wall="HardWall", grid_dr=0.5)
    solver = picard_geometry_solver(cDFT=dft, ng_extrapolations=10, line_search="ERROR")
    #solver.minimise(print_frequency=20,
    #                plot_profile=False)
    solver.animate(z_max=4.0)

    data_dict = get_data_container("../testRF.dat", labels=["RF"], x_index=0, y_indices=[2])
    solver.plot_equilibrium_density_profile(data_dict=data_dict)

    omega, omega_arr = solver.cDFT.grand_potential(solver.density, update_convolutions=True)
    print("omega", omega)