#!/usr/bin/env python3
from dft_numerics import dft_solver
import sys
from constants import NA, KB, Geometry, Specification, LenghtUnit
from grid import Grid, Bulk, Profile
from convolver import Convolver
from pyctp import pcsaft
from pcsaft_functional import pc_saft
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
                 thermopack,
                 temperature,
                 domain_size=100.0,
                 n_grid=1024,
                 specification = Specification.NUMBER_OF_MOLES):
        """Class holding specifications for an interface calculation

        Args:
            geometry (int): PLANAR/POLAR/SPHERICAL
            thermopack (thermo): Thermopack instance
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            specification (Specification, optional): Override how system of equations are solved
        Returns:
            None
        """
        # Create functional
        if isinstance(thermopack, pcsaft.pcsaft):
            t_red = temperature/thermopack.eps_div_kb[0]
            self.functional = pc_saft(n_grid, thermopack, t_red)
        else:
            raise TypeError("No DFT functional for thermopack model: " + type(thermopack))
        # Set up grid
        self.grid = Grid(geometry, domain_size, n_grid)
        # Set defaults
        self.profile = None
        self.converged = False
        self.v_ext = np.zeros((self.functional.nc, self.grid.n_grid))
        self.bulk = None
        self.specification = specification
        self.Ntot = None
        self.do_exp_mu = True
        self.convolver = None
        self.n_tot = None

    def unpack_profile(self, xvec):
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
                 n_grid] = self.profile.densities[ic][:]

        if self.specification == Specification.NUMBER_OF_MOLES:
            # Convolution integrals for densities
            self.convolver.convolve_density_profile(self.profile.densities)
            integrals = self.integrate_df_vext()
            # #denum = np.dot(exp_beta_mu, integrals)
            #print("integrals",integrals)
            exp_mu = self.n_tot / integrals
            #print("exp_mu",exp_mu)
            #self.cDFT.mu_scaled_beta[:] = np.log(0.02276177)
            exp_mu = np.exp(self.bulk.mu_scaled_beta)
            if self.do_exp_mu:
                xvec[n_rho:n_rho + n_c] = exp_mu
            else:
                xvec[n_rho:n_rho + n_c] = np.log(exp_mu)

        return xvec

    def residual(self, xvec):
        # Set profile
        n_grid = self.grid.n_grid
        n_c = self.functional.nc
        prof = self.unpack_profile(xvec)

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

        # Perform convolution integrals
        self.convolver.convolve_density_profile(prof)

        # Calculate new density profile using the variations of the functional
        res = np.zeros(n_rho + n_c *
                       (1 if self.specification ==
                        Specification.NUMBER_OF_MOLES else 0))

        for ic in range(n_c):
            res[ic * n_grid:(ic+1)*n_grid] = - np.exp(self.convolver.correlation(ic)[:]
                       + beta_mu[:] - self.bulk.beta * self.v_ext[ic][:]) \
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
            self.n_tot = self.calculate_total_mass()
            #print("n_tot", self.n_tot)
            # Set up convolver
            self.convolver = Convolver(self.grid, self.functional, self.bulk.R)
            x0 = self.pack_x_vec()
            #print("x0:", x0)
            #sys.exit()
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
        reduced_temperature = min(vle.temperature/t_crit, 1.0)

        self.profile = Profile.tanh_profile(self.grid,
                                            self.bulk,
                                            reduced_temperature,
                                            rel_pos_dividing_surface=rel_pos_dividing_surface)
        return self

    def constant_profile(self, state):
        """
        Initialize constant density profiles. Correct for external potential if present.

        Returns:
            state (State): Thermodynamic state
        """

        # Calculate chemical potential (excess + ideal)
        self.bulk = Bulk(self.functional, state, state)
        self.profile = Profile.constant_profile(self.grid, self.bulk, self.v_ext)
        return self

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
            omega_a[:] += self.T * dens[i][:] * \
                (np.log(dens[i][:]) - 1.0)
            # Extrinsic part
            omega_a[:] += dens[i][:] \
                * (self.v_ext[i][:] - mu[i])

        omega_a[:] *= self.dr

        for i in range(self.nc):
            omega_a[self.boundary_mask[i]] = 0.0  # Don't include wall

        # Integrate
        omega = np.sum(omega_a[:])

        return omega, omega_a

    def surface_tension(self, update_convolutions=True):
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
        for ic in range(self.functional.nc):
            n_tot += np.sum(self.profile.densities[ic][:]*self.grid.integration_weights[:])
        return n_tot

    def integrate_df_vext(self):
        """
        Calculates the integral of exp(-beta(df+Vext)).

        Returns:
            (float): Integral (-)
        """
        n_c = self.functional.nc
        integral = np.zeros(n_c)
        for ic in range(n_c):
            integral[ic] = np.sum(self.grid.integration_weights*
                np.exp(self.convolver.correlation(ic)[:]
                       - self.bulk.beta * self.v_ext[ic][:]))
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
                   np.c_[self.grid.r[:],
                         nd_densities[:, :],
                         (nd_densities[:, :].T / self.cDFT.bulk_densities).T],
                   header="# r, rho, rho/rho_bulk")

    def plot_equilibrium_density_profiles(self,
                                          data_dict=None,
                                          xlim=None,
                                          ylim=None,
                                          plot_actual_densities=False,
                                          plot_equimolar_surface=False,
                                          unit=LenghtUnit.REDUCED):
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
            ax.plot(self.grid.r[:]*len_fac,
                    self.profile.densities[i][:] * dens_fac[i],
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

class PlanarInterface(Interface):
    """
    Utility class for simplifying specification of PLANAR interface
    """

    def __init__(self,
                 thermopack,
                 temperature,
                 domain_size=100.0,
                 n_grid=1024,
                 specification = Specification.NUMBER_OF_MOLES):
        """Class holding specifications for an interface calculation

        Args:
            thermopack (thermo): Thermopack instance
            temperature (float): Temperature (K)
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            specification (Specification, optional): Override how system of equations are solved
        Returns:
            None
        """
        Interface.__init__(self,
                           Geometry.PLANAR,
                           thermopack=thermopack,
                           temperature=temperature,
                           domain_size=domain_size,
                           n_grid=n_grid,
                           specification = specification)

    @staticmethod
    def from_tanh_profile(vle,
                          t_crit,
                          domain_size=100.0,
                          n_grid=1024,
                          rel_pos_dividing_surface=0.5,
                          invert_states=False):
        """
        Initialize tangens hyperbolicus profile

            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface. Default value 0.5.
        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
        """
        pif = PlanarInterface(thermopack=vle.eos,
                              temperature=vle.temperature,
                              domain_size=domain_size,
                              n_grid=n_grid)
        pif.tanh_profile(vle,
                         t_crit,
                         rel_pos_dividing_surface=rel_pos_dividing_surface,
                         invert_states=invert_states)
        return pif

if __name__ == "__main__":
    pass
