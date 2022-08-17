#!/usr/bin/env python3
from dft_numerics import dft_solver
import sys
from constants import NA, KB, Geometry
from weight_functions_cosine_sine import planar_weights_system_mc, \
    planar_weights_system_mc_pc_saft
from utility import packing_fraction_from_density, \
    boundary_condition, densities, get_thermopack_model, \
    weighted_densities_pc_saft_1D, get_initial_densities_vle, \
    weighted_densities_1D
import fmt_functionals
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from grid import Grid

class Interface(object):
    """

    """

    def __init__(self,
                 geometry,
                 functional,
                 domain_size=100.0,
                 n_grid=1024):
        """Class holding specifications for an interface calculation

        Args:
            geometry (int): PLANAR/POLAR/SPHERICAL
            functional (Functional): DFT functional
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.

        Returns:
            None
        """
        self.functional = functional
        self.grid = Grid(geometry, domain_size, n_grid)
        self.profile = None
        self.converged = False
        self.v_ext = None
        self.bulk = None
        self.specification = Specification.NUMBER_OF_MOLES
        self.Ntot = None
        self.do_exp_mu = True

    def unpack_profile(self, x):
        # Set profile
        n_grid = self.grid.n_grid
        n_c = self.functional.nc
        prof = Profile.empty_profile(n_c, n_grid)
        # Make sure boundary cells are set to bulk densities
        #self.mod_densities.assign_elements(self.densities)
        # Calculate weighted densities
        for ic in range(n_c):
            prof.densities[ic][:] = xvec[ic*n_grid:(ic+1)*n_grid]
        return prof

    def pack_x_vec(self):
        n_grid = self.grid.n_grid
        n_c = self.functional.nc
        n_rho = n_c * n_grid
        xvec = np.zeros(n_rho + n_c *
                        (1 if self.specification ==
                         Specification.NUMBER_OF_MOLES else 0))
        for ic in range(n_c):
            xvec[ic * n_grid:(ic+1) *
                 n_grid] = self.profile.densities[ic][self.domain_mask]

        if self.specification == Specification.NUMBER_OF_MOLES:
            # Convolution integrals for densities
            # self.cDFT.weights_system.convolutions(self.mod_densities)
            # # Calculate one-body direct correlation function
            # self.cDFT.weights_system.correlation_convolution()
            # integrals = self.cDFT.integrate_df_vext()
            # #denum = np.dot(exp_beta_mu, integrals)
            # exp_mu = self.Ntot / integrals

            #self.cDFT.mu_scaled_beta[:] = np.log(0.02276177)
            exp_mu = self.mu_scaled_beta
            if self.do_exp_mu:
                xvec[n_rho:n_rho + n_c] = exp_mu
            else:
                xvec[n_rho:n_rho + n_c] = np.log(exp_mu)

        return xvec

    def residual(self, x):
        # Set profile
        n_grid = self.grid.n_grid
        n_c = self.functional.nc
        prof = self.unpack_profile(x)

        n_rho = n_c * n_grid
        beta_mu = np.zeros(n_c)
        if self.specification == Specification.NUMBER_OF_MOLES:
            if self.do_exp_mu:
                exp_beta_mu = np.zeros(n_c)
                exp_beta_mu[:] = xvec[n_rho:n_rho + n_c]
                beta_mu[:] = np.log(exp_beta_mu[:])
                print(beta_mu[:])
            else:
                beta_mu[:] = xvec[n_rho:n_rho + n_c]
                exp_beta_mu = np.zeros(n_c)
                exp_beta_mu[:] = np.exp(beta_mu[:])
            exp_beta_mu_grid = np.zeros(n_c)
            integrals = self.integrate_df_vext()
            #print("integrals",integrals)
            denum = np.dot(exp_beta_mu, integrals)
            exp_beta_mu_grid[:] = self.Ntot * exp_beta_mu / denum
            beta_mu_grid = np.zeros(n_c)
            beta_mu_grid[:] = np.log(exp_beta_mu_grid[:])
            if self.geometry is Geometry.PLANAR:
                # We know the chemical potential - use it
                beta_mu[:] = self.bulk.mu_scaled_beta[:]
        else:
            beta_mu[:] = self.bulk.mu_scaled_beta[:]

        # Calculate new density profile using the variations of the functional
        res = np.zeros(n_rho + n_c *
                       (1 if self.specification ==
                        Specification.NUMBER_OF_MOLES else 0))

        for ic in range(n_c):
            res[ic * n_grid:(ic+1)*n_grid] = - np.exp(self.cDFT.weights_system.comp_differentials[ic].corr[self.domain_mask]
                       + beta_mu[:] - self.cDFT.beta * self.v_ext[ic][self.domain_mask]) \
                + xvec[ic * n_grid:(ic+1)*n_grid]

        if self.specification == Specification.NUMBER_OF_MOLES:
            if self.do_exp_mu:
                res[n_rho:] = exp_beta_mu - exp_beta_mu_grid
            else:
                res[n_rho:] = beta_mu - beta_mu_grid

        return res

    def solve(self, solver=dft_solver(), log_iter=False):
        if not self.profile:
            self.converged = False
            print("Interface need to be initialized before calling solve")
        else:
            x0 = self.pack_x_vec()
            x_sol, self.converged = solver.solve(
                x0, self.residual, log_iter)
            if self.converged:
                self.profile = self.unpack_profile(x_sol)
            else:
                print("Interface solver did not converge")


    def tanh_profile(self, vle, t_crit, rel_pos_dividing_surface=0.5, invert_states=False):
        """
        Initialize tangens hyperbolicus profile

            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface. Default value 0.5.
        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
        """

        left_state = vle.vapor
        right_state = vle.liquid
        if invert_states:
            left_state, right_state = right_state, left_state

        self.bulk = Bulk(self.functional, left_state, right_state)

        # Calculate chemical potential (excess + ideal)
        reduced_temperature = self.min(vle.temperature/t_crit, 1.0)

        self.profile = Profile.tanh_profile(self.grid,
                                            self.bulk,
                                            reduced_temperature,
                                            rel_pos_dividing_surface=rel_pos_dividing_surface)

    def constant_profile(self, state):
        """
        Initialize constant density profiles. Correct for external potential if present.

        Returns:
            state (State): Thermodynamic state
        """

        # Calculate chemical potential (excess + ideal)
        self.bulk = Bulk(self.functional, state, state)
        self.profile = Profile.constant_profile(self.grid, self.bulk, self.v_ext)


    def grand_potential(self):
        """
        Calculates the grand potential in the system.

        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
        """

        # Calculate chemical potential (excess + ideal)
        mu = self.T * (self.mu_res_scaled_beta + np.log(self.bulk_densities))

        # FMT hard-sphere part
        omega_a = self.T * \
            self.functional.excess_free_energy(
                self.weights_system.weighted_densities)

        # Add ideal part and extrinsic part
        for i in range(self.nc):
            # Ideal part
            omega_a[self.domain_mask] += self.T * dens[i][self.domain_mask] * \
                (np.log(dens[i][self.domain_mask]) - 1.0)
            # Extrinsic part
            omega_a[self.domain_mask] += dens[i][self.domain_mask] \
                * (self.Vext[i][self.domain_mask] - mu[i])

        omega_a[:] *= self.dr

        for i in range(self.nc):
            omega_a[self.boundary_mask[i]] = 0.0  # Don't include wall

        # Integrate
        omega = np.sum(omega_a[:])

        return omega, omega_a

    def surface_tension(self, dens, update_convolutions=True):
        """
        Calculates the surface tension of the system.

        Args:
            dens (densities): Density profile
            update_convolutions(bool): Flag telling if convolutions should be updated

        Returns:
            (float): Surface tension
        """

        _, omega_a = self.grand_potential(dens, update_convolutions)
        omega_a += self.red_pressure * self.dr
        for i in range(self.nc):
            omega_a[self.boundary_mask[i]] = 0.0  # Don't include wall

        gamma = np.sum(omega_a)

        return gamma

    def surface_tension_real_units(self, dens, update_convolutions=True):
        """
        Calculates the surface tension of the system.

        Args:
            dens (densities): Density profile
            update_convolutions(bool): Flag telling if convolutions should be updated

        Returns:
            (float): Surface tension (J/m2)
        """

        gamma_star = self.surface_tension(dens, update_convolutions)
        gamma = gamma_star * self.eps / self.sigma ** 2

        return gamma


    def calculate_total_mass(self):
        """
        Calculates the overall mass of the system.

        Args:
            dens (densities): Density profiles

        Returns:
            (float): Surface tension (mol)
        """

        n_tot = 0.0
        for ic in range(self.nc):
            n_tot += np.sum(self.profile.densities[ic][self.domain_mask]*self.grid.integration_weights[:])
        return n_tot

    def integrate_df_vext(self):
        """
        Calculates the integral of exp(-beta(df+Vext)).

        Returns:
            (float): Integral (-)
        """
        n_c = self.functiona.nc
        integral = np.zeros(n_c)
        for ic in range(n_c):
            integral[ic] = np.sum(self.grid.integration_weights*(
                np.exp(self.weights_system.comp_differentials[ic].corr[self.domain_mask]
                       - self.beta * self.Vext[ic][self.domain_mask])))
        return integral


    # def test_laplace(self, dens, sigma0):
    #     """
    #     Calculates the integral of exp(-beta(df+Vext)).

    #     Returns:
    #         (float): Integral (-)
    #     """
    #     #
    #     return 0.0

    def get_equimolar_dividing_surface(self):
        """

        """
        mu = self.bulk.real_mu

        rho_1 = self.profile.densities[i][1]
        rho_1_real = self.bulk.get_real_density(rho_1)
        rho_1_real = self.functinal.thermo.solve_mu_t(self.temperature, mu, rho_initial=rho_1_real)
        rho_1 = self.bulk.get_reduced_density(rho_1_real)
        rho1 = np.sum(rho_1)

        rho_2 = self.profile.densities[i][-2]
        rho_2_real = self.bulk.get_real_density(rho_2)
        rho_2_real = self.functinal.thermo.solve_mu_t(self.temperature, mu, rho_initial=rho_2_real)
        rho_2 = self.bulk.get_reduced_density(rho_2_real)
        rho2 = np.sum(rho_2)

        N = self.calculate_total_mass()
        if self.geometry == Geometry.PLANAR:
            V = self.grid.domain_length
            prefac = 1.0
            exponent = 1.0
        elif self.geometry == Geometry.SPHERICAL:
            prefac = 4*np.pi/3
            V = prefac*self.grid.domain_length**3
            exponent = 1.0/3.0
        elif self.geometry == Geometry.POLAR:
            prefac = np.pi
            V = self.grid.domain_length**2
            exponent = 0.5

        R = ((N - V*rho2)/(rho1 - rho2)/prefac)**exponent
        print("Re, Re/R", R, R/self.grid.domain_length)

        return R

    def print_perform_minimization_message(self):
        """

        """
        print('A successful minimisation have not been yet converged, and the equilibrium profile is missing.')
        print('Please perform a minimisation before performing result operations.')

    def generate_case_name(self):
        """
        Generate case name from specifications
        """
        self.case_name = f'{self.geometry.name}_{"{:.3f}".format(self.bulk.temperature)}_{self.functional.short_name}'

    def save_equilibrium_density_profile(self):
        """
        Save equilibrium density profile to file
        """
        if not self.converged:
            self.print_perform_minimization_message()
            return

        nd_densities = self.profiles.densities.get_nd_copy()
        filename = self.case_name + '.dat'
        np.savetxt(filename,
                   np.c_[self.grid.r[self.domain_mask],
                         nd_densities[:, self.domain_mask],
                         (nd_densities[:, self.domain_mask].T / self.cDFT.bulk_densities).T],
                   header="# r, rho, rho/rho_bulk")

    def plot_equilibrium_density_profiles(self,
                                          data_dict=None,
                                          xlim=None,
                                          ylim=None,
                                          plot_actual_densities=False,
                                          plot_equimolar_surface=False,
                                          unit=Lenght_unit.REDUCED):
        """
        Plot equilibrium density profile
        Args:
            data_dict: Additional data to plot
        """
        if not self.converged:
            self.print_perform_minimization_message()
            return
        fig, ax = plt.subplots(1, 1)
        if unit==Lenght_unit.REDUCED:
            ax.set_xlabel("$z/d_{11}$")
            len_fac = 1.0
        elif unit==Lenght_unit.ANGSTROM:
            ax.set_xlabel("$z$ (Å)")
            len_fac = self.cDFT.sigma*1e10
        des_fac = np.ones(self.profiles.densities.nc)
        if plot_actual_densities:
            des_fac *= 1.0e-3
            ax.set_ylabel(r"$\rho$ (kmol/m$**3$)")
        else:
            des_fac /= self.bulk.bulk_densities
            ax.set_ylabel(r"$\rho^*/\rho_{\rm{b}}^*$")
        for i in range(self.profiles.densities.nc):
            ax.plot(self.grid.r[self.domain_mask]*len_fac,
                    self.profile.densities[i][self.domain_mask] * dens_fac[i],
                    lw=2, color="k", label=f"cDFT comp. {i+1}")
        if data_dict is not None:
            plot_data_container(data_dict, ax)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if plot_equimolar_surface:
            # Plot equimolar dividing surface
            Re = self.get_equimolar_dividing_surface()
            yl = ax.get_ylim()
            ax.plot([Re, Re],
                    [0.0, yl[1]],
                    lw=1, color="k",
                    linestyle="--",
                    label="Eq. mol. surf.")

        leg = plt.legend(loc="best", numpoints=1, frameon=False)

        filename = self.case_name + ".pdf"
        plt.savefig(filename)
        plt.show()

    def grand_potential_bulk(self, wdens, Vext=0.0):
        """
        Calculates the grand potential in the system in bulk.
        Method used for testing.
        Args:
            dens : Weigthed densities
            Vext (float): External potential in bulk

        Returns:
            (float): Grand potential per volume
        """

        # Calculate chemical potential (excess + ideal)
        mu = self.T * (self.mu_res_scaled_beta + np.log(self.bulk_densities))

        # FMT hard-sphere part
        omega_a = self.T * \
            self.functional.excess_free_energy(wdens)

        # Add ideal part and extrinsic part
        for i in range(self.nc):
            # Ideal part
            omega_a[:] += self.T * self.bulk_densities[i] * \
                (np.log(self.bulk_densities[i]) - 1.0)
            # Extrinsic part
            omega_a[:] += self.bulk_densities[i] \
                * (Vext - mu[i])

        return omega_a[0]

    def test_grand_potential_bulk(self):
        """
        """
        # Test grand potential in bulk phase
        wdens = weighted_densities_1D(
            1, self.functional.R, ms=np.ones(self.nc))
        wdens.set_testing_values(rho=self.bulk_densities)
        wdens.n1v[:] = 0.0
        wdens.n2v[:] = 0.0
        omega = self.grand_potential_bulk(wdens, Vext=0.0)
        print("omega:", omega)
        print("pressure + omega:", self.red_pressure + omega)

class Pore(Interface):
    """

    """

    def __init__(self,
                 geometry,
                 functional,
                 external_potential,
                 domain_size=100.0,
                 n_grid=1024):
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
        Returns:
            None
        """

        # Init of base class
        Interface.__init__(self,
                           geometry=geometry,
                           functional=functional,
                           domain_size=domain_size,
                           n_grid=n_grid)
        self.v_ext = external_potential

    def residual(self, x):
        return self.grid.convolve(x).residual


class PairCorrelation(Pore):

    def __init__(self,
                 functional,
                 state,
                 comp=0,
                 domain_size=15.0,
                 n_grid=1024):
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
        Returns:
            None
        """

        v_ext = []
        for i in functional.nc:
            v_ext.append(functional.potential(i, comp, state.T))
        # Init of base class
        Pore.__init__(self,
                           geometry=Geometry.SPHERICAL,
                           functional=functional,
                           external_potential=v_ext,
                           domain_size=domain_size,
                           n_grid=n_grid)


class Surface(Pore):

    def __init__(self,
                 geometry,
                 functional,
                 external_potential,
                 surface_position,
                 domain_size=100.0,
                 n_grid=1024):
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
        Returns:
            None
        """

        # Init of base class
        Pore.__init__(self,
                           geometry=geometry,
                           functional=functional,
                           external_potential=external_potential,
                           domain_size=domain_size,
                           n_grid=n_grid)

   def print_grid(self):
        """
        Debug function

        """
        print("N: ", self.N)
        print("NiWall: ", self.NiWall)
        print("NinP: ", self.NinP)
        print("Nbc: ", self.Nbc)
        print("end: ", self.end)
        print("domain_mask: ", self.domain_mask)
        print("weight_mask: ", self.weight_mask)
        for i in range(self.nc):
            print(f"NiWall_array_left {i}: ", self.NiWall_array_left[i])
            print(f"NiWall_array_right {i}: ", self.NiWall_array_right[i])
            print(f"left_boundary_mask {i}: ", self.left_boundary_mask[i])
            print(f"right_boundary_mask {i}: ", self.right_boundary_mask[i])
            print(f"boundary_mask {i}: ", self.boundary_mask[i])

    def wall_setup(self, wall):
        """

        Args:
            wall (str): Wall type

        """
        self.left_boundary = boundary_condition["OPEN"]
        self.right_boundary = boundary_condition["OPEN"]
        if wall.upper() == "NONE":
            self.wall = "NONE"
        # Wall setup
        hw = ("HW", "HARDWALL", "SHW")
        is_hard_wall = len([w for w in hw if w in wall.upper()]) > 0
        slit = ("SLIT", "SHW")
        is_slit = len([s for s in slit if s in wall.upper()]) > 0
        if is_hard_wall:
            self.left_boundary = boundary_condition["WALL"]
            self.wall = "HW"
            self.weight_mask[:self.NiWall + 1] = True
            for i in range(self.nc):
                self.NiWall_array_left[i] += round(self.NinP[i]/2)
                self.Vext[i][:self.NiWall_array_left[i]] = 500.0
            if is_slit:
                # Add right wall setup
                self.right_boundary = boundary_condition["WALL"]
                self.wall = "SHW"
                self.weight_mask[self.end - 1:] = True
                for i in range(self.nc):
                    self.NiWall_array_right[i] -= round(self.NinP[i] / 2)
                    self.Vext[i][self.NiWall_array_right[i]:] = 500.0

    def get_density_profile(self, density_init, z):
        """

        Args:
            density_init (str): How to initialize density profiles? ("Constant", "VLE")
        Return:
            rho0 (densitiies): Initial density profiles

        """
        if density_init.upper() == "VLE":
            z_centered = np.zeros_like(z)
            z_centered[:] = z[:] - 0.5*(z[0] + z[-1])
            rho0 = get_initial_densities_vle(z_centered,
                                             self.bulk_densities_g,
                                             self.bulk_densities,
                                             self.R,
                                             self.t_div_tc)
        else:
            rho0 = densities(self.nc, self.N)
            rho0.assign_components(self.bulk_densities)

        return rho0

    def grand_potential(self, dens, update_convolutions=True):
        """
        Calculates the grand potential in the system.

        Args:
            dens (densities): Density profile
            update_convolutions(bool): Flag telling if convolutions should be calculated

        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
        """

        # Make sure weighted densities are up-to-date
        if update_convolutions:
            self.weights_system.convolutions(dens)

        # Calculate chemical potential (excess + ideal)
        mu = self.T * (self.mu_res_scaled_beta + np.log(self.bulk_densities))

        # FMT hard-sphere part
        omega_a = self.T * \
            self.functional.excess_free_energy(
                self.weights_system.weighted_densities)

        # Add ideal part and extrinsic part
        for i in range(self.nc):
            # Ideal part
            omega_a[self.domain_mask] += self.T * dens[i][self.domain_mask] * \
                (np.log(dens[i][self.domain_mask]) - 1.0)
            # Extrinsic part
            omega_a[self.domain_mask] += dens[i][self.domain_mask] \
                * (self.Vext[i][self.domain_mask] - mu[i])

        omega_a[:] *= self.dr

        for i in range(self.nc):
            omega_a[self.boundary_mask[i]] = 0.0  # Don't include wall

        # Integrate
        omega = np.sum(omega_a[:])

        return omega, omega_a

    def grand_potential_bulk(self, wdens, Vext=0.0):
        """
        Calculates the grand potential in the system in bulk.
        Method used for testing.
        Args:
            dens : Weigthed densities
            Vext (float): External potential in bulk

        Returns:
            (float): Grand potential per volume
        """

        # Calculate chemical potential (excess + ideal)
        mu = self.T * (self.mu_res_scaled_beta + np.log(self.bulk_densities))

        # FMT hard-sphere part
        omega_a = self.T * \
            self.functional.excess_free_energy(wdens)

        # Add ideal part and extrinsic part
        for i in range(self.nc):
            # Ideal part
            omega_a[:] += self.T * self.bulk_densities[i] * \
                (np.log(self.bulk_densities[i]) - 1.0)
            # Extrinsic part
            omega_a[:] += self.bulk_densities[i] \
                * (Vext - mu[i])

        return omega_a[0]

    def test_grand_potential_bulk(self):
        """
        """
        # Test grand potential in bulk phase
        wdens = weighted_densities_1D(
            1, self.functional.R, ms=np.ones(self.nc))
        wdens.set_testing_values(rho=self.bulk_densities)
        wdens.n1v[:] = 0.0
        wdens.n2v[:] = 0.0
        omega = self.grand_potential_bulk(wdens, Vext=0.0)
        print("omega:", omega)
        print("pressure + omega:", self.red_pressure + omega)


    def calculate_total_mass(self, dens):
        """
        Calculates the overall mass of the system.

        Args:
            dens (densities): Density profiles

        Returns:
            (float): Surface tension (mol)
        """

        n_tot = 0.0
        for ic in range(self.nc):
            n_tot += np.sum(dens[ic])
        n_tot *= self.dr
        return n_tot

    def integrate_df_vext(self):
        """
        Calculates the integral of exp(-beta(df+Vext)).

        Returns:
            (float): Integral (-)
        """
        integral = np.zeros(self.nc)
        for ic in range(self.nc):
            integral[ic] = self.dr*np.sum(np.exp(self.weights_system.comp_differentials[ic].corr[self.domain_mask]
                                                 - self.beta * self.Vext[ic][self.domain_mask]))
        return integral


class cdft_thermopack(cdft1D):
    """
    Base classical DFT class for 1D problems
    """

    def __init__(self,
                 model,
                 comp_names,
                 comp,
                 temperature,
                 pressure,
                 bubble_point_pressure=False,
                 wall="None",
                 domain_length=40.0,
                 grid=1024,
                 phi_disp=1.3862,
                 kwthermoargs={}):
        """
        Object holding specifications for classical DFT problem.
        Reduced particle size assumed to be d=1.0, and all other sizes are relative to this scale.

        Args:
            model (str): Themopack model "PC-SAFT", "SAFT-VR Mie"
            comp_names (str): Component names
            comp (array like): Composition
            temperature (float): Temperature (K)
            pressure (float): Pressure (MPa)
            bubble_point_pressure (bool): Calculate bubble point pressure
            wall (str): Wall type (HardWall, SlitHardWall)
            domain_length (float): Length of domain
            grid (int) : Grid size
            phi_disp (float): Weigthing distance for disperesion term
        Returns:
            None
        """
        self.thermo = get_thermopack_model(model)
        self.thermo.init(comp_names, **kwthermoargs)
        self.thermo.set_tmin(0.75 * temperature)
        self.comp = comp
        if bubble_point_pressure:
            # print(temperature, comp)
            self.eos_pressure, self.eos_gas_comp = self.thermo.bubble_pressure(
                temperature, comp)
            self.eos_liq_comp = self.comp
            self.eos_phase = self.thermo.TWOPH
        else:
            flash = self.thermo.two_phase_tpflash(temperature, pressure, comp)
            self.eos_pressure = pressure
            self.eos_liq_comp = flash[0]
            self.eos_gas_comp = flash[1]
            self.eos_beta_gas = flash[2]
            self.eos_phase = flash[4]

        if self.eos_phase == self.thermo.TWOPH:
            self.eos_vl = self.thermo.specific_volume(temperature,
                                                      self.eos_pressure,
                                                      self.eos_liq_comp,
                                                      self.thermo.LIQPH)
            self.eos_vg = self.thermo.specific_volume(temperature,
                                                      self.eos_pressure,
                                                      self.eos_gas_comp,
                                                      self.thermo.VAPPH)
        else:
            self.eos_vl = self.thermo.specific_volume(temperature,
                                                      self.eos_pressure,
                                                      comp,
                                                      self.eos_phase)
            self.eos_vg = np.ones_like(self.eos_vl)
            self.eos_gas_comp = np.zeros_like(self.eos_vl)

        particle_diameters = np.zeros(self.thermo.nc)
        particle_diameters[:] = self.thermo.hard_sphere_diameters(temperature)
        d_hs_reducing = particle_diameters[0]

        # Store the bulk component densities (scaled)
        self.bulk_densities = np.zeros(self.thermo.nc)
        self.bulk_densities[:] = self.eos_liq_comp[:]/self.eos_vl
        self.bulk_densities[:] *= NA*particle_diameters[0]**3
        self.bulk_densities_g = np.zeros(self.thermo.nc)
        self.bulk_densities_g[:] = self.eos_gas_comp[:]/self.eos_vg
        self.bulk_densities_g[:] *= NA*particle_diameters[0]**3

        # Other quantities
        particle_diameters[:] /= particle_diameters[0]
        temp_red = temperature / self.thermo.eps_div_kb[0]
        grid_dr = domain_length / grid

        # The dispersion term Phi and grid points around that
        # Ø to M: Why is this not a "separate grid?, could end up being
        # just two grid-points with few grid points?
        self.phi_disp = phi_disp
        self.Nbc = 2 * round(self.phi_disp *
                             np.max(particle_diameters) / grid_dr)

        # Calling now the real base class for 1D problems
        cdft1D.__init__(self,
                        bulk_densities=self.bulk_densities,
                        bulk_densities_g=self.bulk_densities_g,
                        particle_diameters=particle_diameters,
                        wall=wall,
                        domain_length=domain_length,
                        functional="PC-SAFT",
                        grid=grid,
                        temperature=temp_red,
                        thermopack=self.thermo)

        # Reduced units
        self.eps = self.thermo.eps_div_kb[0] * KB
        self.sigma = d_hs_reducing

        # Calculate reduced temperature
        Tc, _, _ = self.thermo.critical(comp)
        self.t_div_tc = temperature / Tc

    def test_initial_vle_state(self):
        """
        """
        # Test bulk differentials
        self.functional.test_bulk_differentials(self.bulk_densities)
        z_l = self.functional.bulk_compressibility(self.bulk_densities)
        print("z_l", z_l)
        z_g = self.functional.bulk_compressibility(self.bulk_densities_g)
        print("z_g", z_g)
        mu_l = self.functional.bulk_excess_chemical_potential(
            self.bulk_densities) + np.log(self.bulk_densities)
        mu_g = self.functional.bulk_excess_chemical_potential(
            self.bulk_densities_g) + np.log(self.bulk_densities_g)
        print("mu_l, mu_g, mu_l-mu_g", mu_l, mu_g, mu_l-mu_g)
        P_g = z_g*np.sum(self.bulk_densities_g) * self.T
        P_l = z_l*np.sum(self.bulk_densities) * self.T
        print("P_l, P_g", P_l, P_g, P_l-P_g)

    def test_grand_potential_bulk(self):
        """
        """
        # Test grand potential in bulk phase
        wdens = weighted_densities_pc_saft_1D(
            1, self.functional.R, ms=self.thermo.m)
        wdens.set_testing_values(rho=self.bulk_densities)
        wdens.n1v[:] = 0.0
        wdens.n2v[:] = 0.0
        omega = self.grand_potential_bulk(wdens, Vext=0.0)
        print("omega:", omega)
        print("pressure + omega:", self.red_pressure + omega)


if __name__ == "__main__":
    # cdft_tp = cdft_thermopack(model="PC-SAFT",
    #                           comp_names="C1",
    #                           comp=np.array([1.0]),
    #                           temperature=100.0,
    #                           pressure=0.0,
    #                           bubble_point_pressure=True,
    #                           domain_length=40.0,
    #                           grid_dr=0.001)

    # cdft_tp.test_initial_vle_state()
    # cdft_tp.test_grand_potential_bulk()

    # sys.exit()
    cdft_tp = cdft_thermopack(model="SAFT-VRQ MIE",
                              comp_names="H2",
                              comp=np.array([1.0]),
                              temperature=25.0,
                              pressure=0.0,
                              bubble_point_pressure=True,
                              domain_length=40.0,
                              grid_dr=0.001,
                              kwthermoargs={"feynman_hibbs_order": 1,
                                            "parameter_reference": "AASEN2019-FH1"})

    cdft_tp.test_initial_vle_state()
    cdft_tp.test_grand_potential_bulk()

    cdft_tp.thermo.print_saft_parameters(1)

    sys.exit()
    from utility import density_from_packing_fraction
    d = np.array([1.0, 3.0/5.0])
    bulk_density = density_from_packing_fraction(
        eta=np.array([0.3105, 0.0607]), d=d)
    cdft1 = cdft1D(bulk_densities=bulk_density,
                   particle_diameters=d,
                   domain_length=40.0,
                   functional="Rosenfeld",
                   grid_dr=0.001,
                   temperature=1.0,
                   quadrature="None")
    cdft1.test_grand_potential_bulk()

    d = np.array([1.0])
    bulk_density = density_from_packing_fraction(eta=np.array([0.2]), d=d)
    # bulk_density = np.array([0.74590459])
    cdft2 = cdft1D(bulk_densities=bulk_density,
                   particle_diameters=d,
                   domain_length=40.0,
                   functional="WHITEBEAR",
                   grid_dr=0.001,
                   temperature=0.8664933679930681,
                   quadrature="None")
    cdft2.test_grand_potential_bulk()
