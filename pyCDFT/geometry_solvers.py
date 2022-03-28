#!/usr/bin/env python3
import numpy as np
from cdft import cdft1D
import matplotlib.pyplot as plt
from utility import boundary_condition, density_from_packing_fraction, \
    get_data_container, plot_data_container, \
    quadratic_polynomial, densities
from constants import DEBUG, LCOLORS
import ng_extrapolation
from matplotlib.animation import FuncAnimation
from fmt_functionals import bulk_weighted_densities
import sys


class picard_geometry_solver():
    """
    Base solver class for minimisation objects.
    """

    def __init__(self,
                 cDFT: cdft1D,
                 alpha_min=0.1,
                 alpha_max=0.9,
                 alpha_initial=0.1,
                 n_alpha_initial=1,
                 ng_extrapolations=None,
                 line_search="None"):
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

        # Set radius for plotting
        self.r = np.linspace(-self.NiWall * self.cDFT.dr,
                             (self.cDFT.end - 1) * self.cDFT.dr, self.cDFT.N)

        # Density profiles
        self.densities = densities(self.cDFT.nc, self.cDFT.N)
        self.old_densities = densities(self.cDFT.nc, self.cDFT.N)
        self.new_densities = densities(self.cDFT.nc, self.cDFT.N)
        self.mod_densities = densities(
            self.cDFT.nc, self.cDFT.N, is_conv_var=True)
        self.temp_densities = densities(self.cDFT.nc, self.cDFT.N)

        self.densities.assign_components(self.cDFT.bulk_densities)
        if self.cDFT.left_boundary == boundary_condition["WALL"]:
            self.densities.set_mask(self.cDFT.left_boundary_mask)
        if self.cDFT.right_boundary == boundary_condition["WALL"]:
            self.densities.set_mask(self.cDFT.right_boundary_mask)
        self.old_densities.assign_elements(self.densities)

        # Configure PyFFTW to use multiple threads
        # fftw.config.NUM_THREADS = 2

        # Set state of solver
        self.converged = False

        # Set state of solver
        self.iteration = 0
        self.error = np.ones(self.cDFT.nc)

        # Set Picard parameter for relaxed successive substitution
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_initial = alpha_initial
        self.n_alpha_initial = n_alpha_initial

        # Error norm (np.inf: Max norm, None: 2 norm, ....)
        self.norm = np.inf

        # Set case name to be used for output files etc.
        self.generate_case_name()

        # Extrapolations according to Ng 1974
        self.ng = ng_extrapolation.ng_nc_wrapper(self.cDFT.nc,
                                                 self.cDFT.N,
                                                 ng_extrapolations,
                                                 self.domain_mask)

        # Line search
        self.do_line_search = not line_search.upper() == "NONE"
        self.line_search_eval = line_search.upper()
        self.ls_densities = densities(self.cDFT.nc, self.cDFT.N)

        # Solver defaults
        self.tolerance = 1.0e-12
        self.print_frequency = 50
        self.maximum_iterations = 10000000

    def picard_density(self, mix_densities, alpha, new_densities=None, old_densities=None):
        """
        Mix new_density and density
        Args:
            alpha (float): Fraction of new density
            mix_densities (densities): Array to be mixed (output)
            new_densities (densities): New density
            old_densities (densities): Old density

        """
        if new_densities is None:
            new_densities = self.new_densities
        if old_densities is None:
            old_densities = self.densities

        for i in range(self.cDFT.nc):
            mix_densities[i][self.cDFT.domain_mask] = (1 - alpha) * \
                old_densities[i][self.cDFT.domain_mask] + \
                alpha * new_densities[i][self.cDFT.domain_mask]

    def successive_substitution(self, dens):
        """
        Perform one successive substitution iteration on the equation system.
        Updates self.new_densities.

        Args:
            dens (densities): Density profiles
        """

        # Calculate weighted densities
        self.mod_densities.assign_elements(dens)
        # self.wall_update()
        # Convolution integrals for densities
        self.cDFT.weights_system.convolutions(self.mod_densities)
        # Calculate one-body direct correlation function
        self.cDFT.weights_system.correlation_convolution()

        # index = 30000
        # print(self.mod_densities[0][index], self.mod_densities[1][index])
        # self.cDFT.weights_system.comp_weighted_densities[0].print(index=index)
        # self.cDFT.weights_system.comp_weighted_densities[1].print(index=index)
        # self.cDFT.weights_system.weighted_densities.print(index=index)
        # bd = bulk_weighted_densities(self.cDFT.bulk_densities, self.cDFT.R)
        # bd.print()
        # sys.exit()

        # Calculate new density profile using the variations of the functional
        for i in range(self.cDFT.nc):
            self.new_densities[i][self.domain_mask] = self.cDFT.bulk_densities[i] * \
                np.exp(self.cDFT.weights_system.differentials[i].corr[self.domain_mask]
                       + self.cDFT.beta *
                       (self.cDFT.excess_mu[i] - self.cDFT.Vext[i][self.domain_mask]))
        if DEBUG:
            print("new_density", self.new_densities)

    def picard_update(self):
        """
        Updates density profile using Picard procedures.

        Returns:
            have_failed (bool): Fail indicator
        """
        # Do successive substitution to get new_density
        self.successive_substitution(self.densities)

        # Check for valid floats
        if not self.new_densities.is_valid_reals():
            have_failed = True
            return have_failed
        else:
            have_failed = False

        # Set old density profile
        self.old_densities.assign_elements(self.densities)

        # Update Ng history
        self.ng.push_back(self.densities, self.new_densities, self.iteration)

        if np.max(self.error) < 1.0e-3 and \
                self.ng.time_to_update(self.iteration) and \
                self.iteration > self.n_alpha_initial:
            self.densities.assign_elements(
                self.ng.extrapolate(), self.domain_mask)
        else:
            # Picard update
            if self.iteration <= self.n_alpha_initial:
                alpha = self.alpha_initial
            else:
                # Do line search?
                if self.do_line_search:
                    # self.plot_line_search()
                    # sys.exit()
                    alpha = min(self.line_search(debug=False), self.alpha_max)
                    # print(alpha)
                else:
                    alpha = self.alpha_min
            self.picard_density(self.densities, alpha)

        # Calculate deviation between new and old density profiles
        self.error[:] = self.densities.diff_norms(
            self.old_densities, ord=self.norm)[:]
        return have_failed

    def iteration_plot_profile(self, plot_old_profile=False):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("$z/\sigma_1$")
        ax.set_ylabel(r"$\rho^*/\rho_{\rm{b}}^*$")
        for i in range(self.densities.nc):
            ax.plot(self.r[self.domain_mask],
                    self.densities[i][self.domain_mask] /
                    self.cDFT.bulk_densities[i],
                    lw=2, color=LCOLORS[i], label=f"Comp. {i+1}")
        if plot_old_profile:
            for i in range(self.densities.nc):
                ax.plot(self.r[self.domain_mask],
                        self.old_densities[i][self.domain_mask] /
                        self.cDFT.bulk_densities[i],
                        lw=2, color=LCOLORS[i], ls="--",
                        label=f"Old. Comp. {i+1}")
        leg = plt.legend(loc="best", numpoints=1)
        leg.get_frame().set_linewidth(0.0)
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
        for _ in range(maximum_iterations):
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

        Returns:
            bool: Did iteration fail?
        """

        have_failed = self.picard_update()
        self.iteration += 1
        if np.max(self.error) < self.tolerance:
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
        Animate iterations when solving roe the equilibrium density profiles.

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
        anim_solver(self, z_max=z_max)
        plt.show()

    def wall_update(self):
        """
        Reduce hard wall density by 50%
        """

        # todo find explanation for 0.5 trick
        if self.cDFT.wall == "HW" or self.cDFT.wall == "SHW":
            for i in range(self.cDFT.nc):
                if self.cDFT.left_boundary == boundary_condition["WALL"]:
                    self.mod_densities[i][self.cDFT.NiWall_array_left[i] + 1] *= 0.5
                if self.cDFT.right_boundary == boundary_condition["WALL"]:
                    self.mod_densities[i][self.cDFT.NiWall_array_right[i] - 1] *= 0.5

    def print_perform_minimization_message(self):
        """

        """
        print('A successful minimisation have not been yet converged, and the equilibrium profile is missing.')
        print('Please perform a minimisation before performing result operations.')

    def generate_case_name(self):
        """
        Generate case name from specifications
        """
        etas = ""
        for i in range(self.cDFT.nc):
            etas += "{:.3f}".format(self.cDFT.eta[i])
            if i < self.cDFT.nc - 1:
                etas += "_"
        self.case_name = f'Planar_{self.cDFT.wall}_{etas}'
        self.case_name += '_' + self.cDFT.functional.short_name

    def save_equilibrium_density_profile(self):
        """
        Save equilibrium density profile to file
        """
        if not self.converged:
            self.print_perform_minimization_message()
            return

        nd_densities = self.densities.get_nd_copy()
        filename = self.case_name + '.dat'
        np.savetxt(filename,
                   np.c_[self.r[self.domain_mask],
                         nd_densities[:, self.domain_mask],
                         (nd_densities[:, self.domain_mask].T / self.cDFT.bulk_densities).T],
                   header="# r, rho, rho/rho_bulk")

    def plot_equilibrium_density_profiles(self, data_dict=None):
        """
        Plot equilibrium density profile
        Args:
            data_dict: Additional data to plot
        """
        if not self.converged:
            self.print_perform_minimization_message()
            return

        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("$z/\sigma_1$")
        ax.set_ylabel(r"$\rho^*/\rho_{\rm{b}}^*$")
        for i in range(self.densities.nc):
            ax.plot(self.r[self.domain_mask],
                    self.densities[i][self.domain_mask] /
                    self.cDFT.bulk_densities[i],
                    lw=2, color="k", label=f"cDFT comp. {i+1}")
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
            alpha
            update=True
        Returns:
            None
        """
        if update:
            self.picard_density(self.temp_densities, alpha,
                                new_densities=self.ls_densities)
        if self.line_search_eval == "ERROR":
            # Knepley et al. 2010. doi: 10.1063/1.3357981
            if update:
                self.successive_substitution(self.temp_densities)
            err = self.new_densities.diff_norm_scaled(other=self.temp_densities,
                                                      scale=self.cDFT.bulk_densities,
                                                      mask=self.domain_mask,
                                                      ord=2)**2
        elif self.line_search_eval.strip("-_") in ["GP", "GRANDPOTENTIAL"]:
            err = self.cDFT.grand_potential(
                self.temp_densities, update_convolutions=update)[0]
        else:
            raise ValueError(
                "get_line_search_error: Wrong type for function evaluation")
        return err

    def calc_alpha_max(self):
        """
        Calculate maximum alpha value such that n3 < 1.0
        See: Knepley et al. 2010. doi: 10.1063/1.3357981

        Returns:
            alpha_max (float): Maximum alpha value
        """
        n3_inf_0 = np.linalg.norm(
            self.cDFT.weights_system.weighted_densities.n3, ord=np.inf)
        self.mod_densities.assign_elements(self.new_densities)
        self.cDFT.weights_system.convolution_n3(self.mod_densities)
        n3_inf_1 = np.linalg.norm(
            self.cDFT.weights_system.weighted_densities.n3, ord=np.inf)
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
        self.ls_densities.assign_elements(self.new_densities)
        # Set to avoid update in get_line_search_error
        self.temp_densities.assign_elements(self.densities)
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
        self.new_densities.assign_elements(self.ls_densities)
        return alpha

    def plot_line_search(self):
        """
        Calculate function values and plot together with quadratic line search polynomial

        """
        alpha_max = self.calc_alpha_max()
        self.ls_densities.assign_elements(self.new_densities)
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
        plt.show()
        # Reset density
        self.new_densities.assign_elements(self.ls_densities)

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
            if have_failed:
                print(f"Simulation failed")
            if self.solver.converged:
                print(
                    f"Simulation converged after {solver.iteration} iterations")
            self.ani.event_source.stop()
        for i in range(self.solver.cDFT.nc):
            self.line[i].set_data(self.solver.r[self.solver.domain_mask],
                                  self.solver.densities[i][self.solver.domain_mask] /
                                  self.solver.cDFT.bulk_densities[i])
        # self.line, = self.ax.plot(self.solver.r[self.solver.domain_mask],
        #                          self.solver.density[self.solver.domain_mask] / self.solver.cDFT.bulk_density,
        #                          lw=2, color="k")
        self.ax.relim()
        self.ax.autoscale_view()
        if self.z_max is not None:
            self.ax.set_xlim(0, self.z_max)
        #self.ax.set_ylim(-1, 1)
        return self.line,

    def init_plot(self):
        self.line = []
        for i in range(self.solver.cDFT.nc):
            line, = self.ax.plot(self.solver.r[self.solver.domain_mask],
                                 self.solver.densities[i][self.solver.domain_mask] /
                                 self.solver.cDFT.bulk_densities[i],
                                 lw=2, color=LCOLORS[i])
            self.line.append(line)
        self.ax.set_xlabel("$z/\sigma_1$")
        self.ax.set_ylabel(r"$\rho^*/\rho_{\rm{b}}^*$")
        return self.line,


if __name__ == "__main__":
    # Binary hard-sphere case from: 10.1103/physreve.62.6926
    d = np.array([1.0, 3.0/5.0])
    bulk_densities = density_from_packing_fraction(
        eta=np.array([0.3105, 0.0607]), d=d)
    dft = cdft1D(bulk_densities=bulk_densities, particle_diameters=d, functional="WB",
                 domain_length=50.0, wall="HardWall", grid_dr=0.001)
    # dft = cdft1D(bulk_densities=bulk_densities, particle_diameters=d, functional="WB",
    #              domain_length=50.0, wall="SlitHardWall", grid_dr=0.001)
    solver = picard_geometry_solver(
        cDFT=dft, alpha_min=0.05, alpha_max=0.25, alpha_initial=0.001, n_alpha_initial=50,
        ng_extrapolations=10, line_search="ERROR")
    solver.minimise(print_frequency=50,
                    plot_profile=True)
    # solver.animate(z_max=4.0)
    sys.exit()
    # Pure hard-sphere case from. Packing fraction 0.2.
    bulk_densities = density_from_packing_fraction(
        eta=np.array([0.2]))
    dft = cdft1D(bulk_densities=bulk_densities, functional="WB",
                 domain_length=50.0, wall="HardWall", grid_dr=0.001)
    # dft = cdft1D(bulk_densities=bulk_densities, domain_length=50.0,
    #              wall="SlitHardWall", grid_dr=0.001)
    # dft = cdft1D(bulk_densities=bulk_densities,
    #              domain_length=3.0, wall="HardWall", grid_dr=0.25, quadrature="NONE")
    solver = picard_geometry_solver(
        cDFT=dft, alpha_min=0.1, ng_extrapolations=10, line_search="ERROR")
    solver.minimise(print_frequency=10,
                    plot_profile=True)
    # solver.animate(z_max=4.0)

    # data_dict = get_data_container(
    #     "../testRF.dat", labels=["RF"], x_index=0, y_indices=[2])
    # solver.plot_equilibrium_density_profiles(data_dict=data_dict)

    omega, omega_arr = solver.cDFT.grand_potential(
        solver.densities, update_convolutions=True)
    print("omega", omega)

    sys.exit()
