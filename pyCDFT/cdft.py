#!/usr/bin/env python3
import numpy as np
import fmt_functionals
from utility import packing_fraction_from_density, \
    boundary_condition, densities, get_thermopack_model, \
    weighted_densities_pc_saft_1D, get_initial_densities_vle, \
    weighted_densities_1D
from weight_functions import planar_weights_system_mc, \
    planar_weights_system_mc_pc_saft
from constants import CONV_FFTW, CONV_SCIPY_FFT, CONV_NO_FFT, CONVOLUTIONS, NA, KB
import sys


class cdft1D:
    """
    Base classical DFT class for 1D problems
    """

    def __init__(self,
                 bulk_densities,
                 bulk_densities_g=None,
                 particle_diameters=np.array([1.0]),
                 wall="HW",
                 domain_length=40.0,
                 functional="Rosenfeld",
                 grid_dr=0.001,
                 temperature=1.0,
                 quadrature="None",
                 thermopack=None):
        """
        Object holding specifications for classical DFT problem.
        Reduced particle size assumed to be d=1.0, and all other sizes are relative to this scale.

        Args:
            bulk_densities (ndarray): Bulk fluid density ()
            particle_diameters (ndarray): Particle diameter
            wall (str): Wall type (HardWall, SlitHardWall, None)
            domain_length (float): Length of domain
            functional (str): Name of hard sphere functional: Rosenfeld, WhiteBear, WhiteBear Mark II, Default Rosenfeld
            grid_dr (float) : Grid spacing
            temperature (float): Reduced temperature
            quadrature (str): Quadrature to use during integration
        Returns:
            None
        """
        # Thermopack
        self.thermo = thermopack
        # Reduced unit
        self.eps = 1.0
        self.sigma = 1.0
        # Number of components
        self.nc = len(particle_diameters)
        # Particle radius
        self.R = 0.5*particle_diameters
        # Temperature
        self.T = temperature
        self.beta = 1.0 / temperature
        # Bulk density
        self.bulk_densities = bulk_densities
        if bulk_densities_g is None:
            self.bulk_densities_g = bulk_densities
        else:
            self.bulk_densities_g = bulk_densities_g
        self.eta = packing_fraction_from_density(
            bulk_densities, d=particle_diameters)
        # Bulk fractions
        self.bulk_fractions = bulk_densities/np.sum(bulk_densities)
        # Length
        self.domain_length = domain_length
        # Grid spacing
        self.dr = grid_dr

        # FFT padding of grid
        if CONVOLUTIONS in (CONV_FFTW, CONV_SCIPY_FFT):
            self.padding = 1
        else:
            self.padding = 0
        # Get grid info
        self.N = round(domain_length / grid_dr)  # Should be even
        self.NinP = []
        for i in range(self.nc):
            # Number of grid points within particle
            self.NinP.append(2 * round(self.R[i] / grid_dr))

        # Test if self.Nbc has been set by child class
        try:
            self.Nbc
        except AttributeError:
            self.Nbc = 0
        self.Nbc = max(self.Nbc, np.max(self.NinP))
        self.padding *= self.Nbc
        # Add boundary and padding to grid
        self.N = self.N + 2 * self.Nbc + 2 * self.padding
        self.end = self.N - self.Nbc - self.padding  # End of domain

        # Get functional
        self.functional = fmt_functionals.get_functional(self.N,
                                                         self.T,
                                                         functional,
                                                         self.R,
                                                         self.thermo)

        # Calculate reduced pressure and excess chemical potential
        self.red_pressure = np.sum(self.bulk_densities) * self.T * \
            self.functional.bulk_compressibility(self.bulk_densities)
        self.excess_mu = self.functional.bulk_excess_chemical_potential(
            self.bulk_densities)
        #print(self.red_pressure, self.excess_mu)

        # Mask for inner domain
        self.NiWall = self.N - self.end
        self.domain_mask = np.full(self.N, False, dtype=bool)
        self.domain_mask[self.NiWall:self.end] = True
        self.weight_mask = np.full(self.N, False, dtype=bool)

        # Set up wall
        self.NiWall_array_left = [self.NiWall] * self.nc
        self.NiWall_array_right = [self.N - self.NiWall] * self.nc
        # Use structure of densities class
        self.Vext = densities(self.nc, self.N)
        self.left_boundary = None  # Handled in setup_wall
        self.right_boundary = None  # Handled in setup_wall
        self.wall = None  # Handled in setup_wall
        self.wall_setup(wall)

        self.left_boundary_mask = []
        self.right_boundary_mask = []
        self.boundary_mask = []
        for i in range(self.nc):
            self.left_boundary_mask.append(
                np.full(self.N, False, dtype=bool))
            self.right_boundary_mask.append(
                np.full(self.N, False, dtype=bool))
            self.boundary_mask.append(
                np.full(self.N, False, dtype=bool))
            self.left_boundary_mask[i][:self.NiWall_array_left[i]] = True
            self.right_boundary_mask[i][self.NiWall_array_right[i]:] = True
            self.boundary_mask[i] = np.logical_or(
                self.left_boundary_mask[i], self.right_boundary_mask[i])

        # self.print_grid()

        # Allocate weighted densities, differentials container and weights
        if functional.upper() in ("PC-SAFT", "PCSAFT"):
            self.weights_system = \
                planar_weights_system_mc_pc_saft(functional=self.functional,
                                                 dr=self.dr,
                                                 R=self.R,
                                                 N=self.N,
                                                 pcsaft=self.thermo,
                                                 mask_conv_results=self.weight_mask)
        else:
            self.weights_system = \
                planar_weights_system_mc(functional=self.functional,
                                         dr=self.dr,
                                         R=self.R,
                                         N=self.N,
                                         quad=quadrature,
                                         mask_conv_results=self.weight_mask)

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
                                             self.R)
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
        mu = self.T * (self.excess_mu + np.log(self.bulk_densities))

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
        mu = self.T * (self.excess_mu + np.log(self.bulk_densities))

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
                 grid_dr=0.001,
                 phi_disp=1.3862):
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
            grid_dr (float) : Grid spacing
            phi_disp (float): Weigthing distance for disperesion term
        Returns:
            None
        """
        self.thermo = get_thermopack_model(model)
        self.thermo.init(comp_names)
        self.comp = comp
        if bubble_point_pressure:
            #print(temperature, comp)
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
        #print(self.eos_vg, self.eos_gas_comp)
        particle_diameters = np.zeros(self.thermo.nc)
        particle_diameters[:] = self.thermo.hard_sphere_diameters(temperature)
        d_hs_reducing = particle_diameters[0]
        self.bulk_densities = np.zeros(self.thermo.nc)
        self.bulk_densities[:] = self.eos_liq_comp[:]/self.eos_vl
        self.bulk_densities[:] *= NA*particle_diameters[0]**3
        self.bulk_densities_g = np.zeros(self.thermo.nc)
        self.bulk_densities_g[:] = self.eos_gas_comp[:]/self.eos_vg
        self.bulk_densities_g[:] *= NA*particle_diameters[0]**3
        particle_diameters[:] /= particle_diameters[0]
        temp_red = temperature / self.thermo.eps_div_kb[0]
        # print(self.bulk_densities_g)

        self.phi_disp = phi_disp
        self.Nbc = 2 * round(self.phi_disp *
                             np.max(particle_diameters) / grid_dr)
        cdft1D.__init__(self,
                        bulk_densities=self.bulk_densities,
                        bulk_densities_g=self.bulk_densities_g,
                        particle_diameters=particle_diameters,
                        wall=wall,
                        domain_length=domain_length,
                        functional=model,
                        grid_dr=grid_dr,
                        temperature=temp_red,
                        thermopack=self.thermo)

        # Reduced unit
        self.eps = self.thermo.eps_div_kb[0] * KB
        self.sigma = d_hs_reducing

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
    cdft_tp = cdft_thermopack(model="PC-SAFT",
                              comp_names="C1",
                              comp=np.array([1.0]),
                              temperature=130.0,
                              pressure=0.0,
                              bubble_point_pressure=True,
                              domain_length=40.0,
                              grid_dr=0.001)
    cdft_tp.test_initial_vle_state()
    cdft_tp.test_grand_potential_bulk()

    # sys.exit()
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
    #bulk_density = np.array([0.74590459])
    cdft2 = cdft1D(bulk_densities=bulk_density,
                   particle_diameters=d,
                   domain_length=40.0,
                   functional="WHITEBEAR",
                   grid_dr=0.001,
                   temperature=0.8664933679930681,
                   quadrature="None")
    cdft2.test_grand_potential_bulk()
