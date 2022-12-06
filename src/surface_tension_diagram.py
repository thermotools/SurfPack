#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from dft_numerics import dft_solver
from constants import Geometry, ProfileInit, LenghtUnit, Specification, Properties, LCOLORS, KB
from interface import PlanarInterface, SphericalInterface
import numpy as np
import matplotlib.pyplot as plt
from pyctp.thermopack_state import MetaCurve, Equilibrium, State
from density_profile import Profile, ProfilePlotter
from matplotlib.animation import FuncAnimation
from abc import ABC, abstractmethod
from tqdm import tqdm
from scipy.interpolate import interp1d

class InterfaceList(ABC):
    """

    """

    def __init__(self):
        """Class for calculating surface tension along saturation curve

        Args:
            curve (PhaseDiagram): List of states to calculate surface tension
            geometry (int): PLANAR/POLAR/SPHERICAL
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            solver (dft_solver, optional): Solver for DFT
            init_profiles (ProfileInit): How to initialize profiles? Default: ProfileInit.TANH
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """
        self.interfaces = []
        self.is_liquid_right = True

    def animate(self,
                z_max=None,
                prop=Properties.RHO,
                plot_reduced_property=True,
                plot_equimolar_surface=False,
                plot_bulk=False,
                include_legend=False,
                filename=None,
                **kw):
        """Class for animating properties

        Args:

        Returns:
            None
        """
        if z_max is not None:
            interfaces = []
            for interface in self.interfaces:
                if interface.r_equimolar < z_max:
                    interfaces.append(interface)
            if plot_reduced_property:
                sigma = self.interfaces[0].functional.thermo.sigma[0]
                z_max = z_max*self.interfaces[0].grid.domain_size/sigma
        else:
            interfaces = self.interfaces

        self.anim = plot_animator(interfaces,
                                  z_max,
                                  prop,
                                  plot_reduced_property,
                                  plot_equimolar_surface,
                                  plot_bulk,
                                  include_legend,
                                  is_liquid_right=self.is_liquid_right,
                                  **kw)
        plt.show()
        if filename is not None:
            self.anim.save(filename)

class SurfaceTensionDiagram(InterfaceList):
    """

    """

    def __init__(self,
                 curve,
                 geometry=Geometry.PLANAR,
                 domain_size=200.0,
                 n_grid=1024,
                 solver=None,
                 init_profiles=ProfileInit.TANH,
                 functional_kwargs={}):
        """Class for calculating surface tension along saturation curve

        Args:
            curve (PhaseDiagram): List of states to calculate surface tension
            geometry (int): PLANAR/POLAR/SPHERICAL
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            solver (dft_solver, optional): Solver for DFT
            init_profiles (ProfileInit): How to initialize profiles? Default: ProfileInit.TANH
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """
        InterfaceList.__init__(self)
        self.surface_tension = []
        self.surface_tension_reduced = []

        #
        t_crit, _, _ = curve.vle_states[0].vapor.eos.critical(curve.vle_states[0].vapor.x)
        #if geometry == Geometry.PLANAR:
        for state in curve.vle_states:
            if (abs(t_crit - state.vapor.T) > 1.0e-4):
                interf = PlanarInterface.from_tanh_profile(state,
                                                           t_crit,
                                                           domain_size=domain_size,
                                                           n_grid=n_grid,
                                                           functional_kwargs=functional_kwargs)
                # Solve for equilibrium profile
                interf.solve(log_iter=False)
                # interf.plot_equilibrium_density_profiles(plot_reduced_densities=True,
                #                                          plot_equimolar_surface=True,
                #                                          grid_unit=LenghtUnit.REDUCED)
                st = interf.surface_tension(reduced_unit=True)
                st_r = interf.surface_tension_real_units()
                self.interfaces.append(interf)
            else:
                st = 0.0
                st_r = 0.0
            self.surface_tension_reduced.append(st)
            self.surface_tension.append(st_r)

            # from_profile(vle,
            #          profile,
            #          domain_size=100.0,
            #          n_grid=1024,
            #          invert_states=False,
            #          functional_kwargs={})

        self.surface_tension_reduced = np.array(self.surface_tension_reduced)
        self.surface_tension = np.array(self.surface_tension)

class SphericalDiagram(InterfaceList):
    """

    """

    def __init__(self,
                 vle,
                 n_steps=1000,
                 n_grid=1024,
                 solver=None,
                 calculate_bubble=True,
                 init_profiles=ProfileInit.TANH,
                 log_iter=False,
                 terminate_betawf=0.1,
                 functional_kwargs={}):
        """Class for calculating surface tension along saturation curve

        Args:
            vle (Equilibrium): Equilibrium state
            n_steps (int): Number of steps for mapping droplets/bubbles. Defaults to 1000.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            solver (dft_solver, optional): Solver for DFT
            calculate_bubble (bool, optional): Calculate bubble? (or droplet?) Defaults to True.
            init_profiles (ProfileInit): How to initialize profiles? Default: ProfileInit.TANH
            terminate_betawf (float, optional): Therminate when reduced work of formation drops below this value. Defaults to 0.1.
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """
        InterfaceList.__init__(self)

        if solver is None:
            solver=dft_solver().anderson(mmax=50, beta=0.05, tolerance=1.0e-12,max_iter=500)

        # Backup solver
        solver2=dft_solver().picard(tolerance=1.0e-8,max_iter=800,beta=0.05,ng_frequency=None).\
            anderson(mmax=50, beta=0.05, tolerance=1.0e-12,max_iter=500)

        # Solve for planar interface
        eos = vle.liquid.eos
        t_crit, _, _ = eos.critical(vle.liquid.x)
        invert_states = False if calculate_bubble else True
        self.planar_interf = PlanarInterface.from_tanh_profile(vle,
                                                               t_crit,
                                                               domain_size=200.0,
                                                               n_grid=n_grid,
                                                               invert_states=invert_states,
                                                               functional_kwargs=functional_kwargs)
        self.planar_interf.solve()
        sigma0 = self.planar_interf.surface_tension(reduced_unit=False)

        self.is_liquid_right = calculate_bubble
        self.phase = eos.LIQPH if calculate_bubble else eos.VAPPH
        self.is_liquid_bulk = calculate_bubble
        self.z = vle.liquid.x if calculate_bubble else vle.vapor.x
        states = MetaCurve.isothermal(eos, vle.temperature, self.z, n_steps, self.phase)

        # Loop large spheres specifiyng particle numbers
        radius_list = np.linspace(50.0, 1000.0, 39)
        real_radius = -1000E-10

        n_grid_large = n_grid*4
        shift=900.0
        grid = self.planar_interf.grid
        bulk = self.planar_interf.bulk
        spi = self.planar_interf
        for r in tqdm(reversed(radius_list), desc="Looping large radius spheres", total=np.shape(radius_list)[0]):
            sr = r * (-1.0 if calculate_bubble else 1.0)*1e-10
            # Extrapolate chemical potential to first order and solve for phase densiteis
            mu, rho_l, rho_g = \
                eos.extrapolate_mu_in_inverse_radius(sigma_0=sigma0,
                                                     temp=vle.temperature,
                                                     rho_l=vle.liquid.rho,
                                                     rho_g=vle.vapor.rho,
                                                     radius=sr,
                                                     geometry="SPHERICAL",
                                                     phase=self.phase)
            vapor = State(eos=vle.eos, T=vle.temperature, V=1/sum(rho_g), n=rho_g/sum(rho_g))
            liquid = State(eos=vle.eos, T=vle.temperature, V=1/sum(rho_l), n=rho_l/sum(rho_l))
            left_state = vapor if calculate_bubble else liquid
            right_state = liquid if calculate_bubble else vapor
            meta = equilibrium(left_state, right_state)
            profile = Profile()
            profile.copy_profile(spi.profile)
            z_new, r_domain = profile.shift_and_scale(shift=shift,
                                                      grid=grid,
                                                      n_grid=n_grid_large,
                                                      interpolation="cubic",
                                                      rho_left=bulk.get_reduced_density(left_state.partial_density()),
                                                      rho_right=bulk.get_reduced_density(right_state.partial_density()))

            spi = SphericalInterface.from_profile(meta,
                                                  profile,
                                                  domain_radius=r_domain,
                                                  n_grid=n_grid_large,
                                                  invert_states=False,
                                                  specification=Specification.NUMBER_OF_MOLES)
            spi.solve(solver=solver,log_iter=log_iter)
            shift=-25.0
            grid = spi.grid

            if spi.converged:
                self.interfaces.append(spi)

        # Decrease grid
        profile = Profile()
        profile.copy_profile(spi.profile)
        z_new, r_domain = profile.shift_and_scale(shift=0.0,
                                                  grid=grid,
                                                  n_grid=n_grid,
                                                  interpolation="cubic")
        spi = SphericalInterface.from_profile(meta,
                                              profile,
                                              domain_radius=r_domain,
                                              n_grid=n_grid,
                                              invert_states=False,
                                              specification=Specification.NUMBER_OF_MOLES)
        #print("Grid reduction")
        spi.solve(solver=solver,log_iter=log_iter)
        #print(spi.r_equimolar)


        # Locate meta-state with reduced bubble/droplet size
        mu_sol = spi.bulk.left_state.chemical_potential()
        loop_meta_states = []
        for i, meta in enumerate(states.meta_states):
            mu_isotherm = meta.liquid.chemical_potential()
            if calculate_bubble:
                if mu_isotherm >= mu_sol:
                    continue
            else:
                if mu_isotherm <= mu_sol:
                    continue
            loop_meta_states.append(meta)

        for meta in tqdm(loop_meta_states, desc="Looping meta-stable states"):
            profile = Profile()
            profile.copy_profile(spi.profile)
            r_ext = self.extrapolate_r_equimolar(meta, n_points=10, interpolation="cubic")

            #left_state = vapor if calculate_bubble else liquid
            #right_state = liquid if calculate_bubble else vapor
            #meta = equilibrium(left_state, right_state)
            profile = Profile()
            profile.copy_profile(spi.profile)
            shift = r_ext - spi.r_equimolar
            print("shift",shift)
            z_new, r_d = profile.shift_and_scale(shift=shift,
                                                 grid=spi.grid,
                                                 n_grid=n_grid,
                                                 interpolation="cubic")
#                                                      rho_left=bulk.get_reduced_density(left_state.partial_density()),
#                                                      rho_right=bulk.get_reduced_density(right_state.partial_density()))

            spi = SphericalInterface.from_profile(meta,
                                                  profile,
                                                  domain_radius=r_domain,
                                                  n_grid=n_grid,
                                                  invert_states=invert_states,
                                                  specification=Specification.CHEMICHAL_POTENTIAL,
                                                  functional_kwargs=functional_kwargs)
            spi.solve(solver=solver, log_iter=log_iter)
            print("N", spi.n_iter)
            print("r: ",r_ext, spi.r_equimolar,r_ext-spi.r_equimolar)
            #spi.plot_property_profiles(plot_equimolar_surface=True,
            #                           plot_bulk=True)

            if not spi.converged:
                spi.solve(solver=solver2, log_iter=log_iter)
            #spi.plot_property_profiles(plot_equimolar_surface=True,
            #                           plot_bulk=True)
            if spi.converged:
                self.interfaces.append(spi)
                if spi.work_of_formation() < terminate_betawf:
                    break

    def extrapolate_r_equimolar(self, meta, n_points=10, interpolation="cubic"):
        """Class for extrapolating equimolar radius

        Args:
            meta (Equilibrium): Meta-stable state at same chemical potential
            n_points (int, optional): Number of points used for extrapolation. Defaults to 10.
            interpolation (str): Interpolation option: ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’
        Returns:
            (float): Extrapolated equimolar radius
        """
        left_state = meta.vapor if self.phase == meta.eos.LIQPH else meta.liquid
        right_state = meta.liquid if self.phase == meta.eos.LIQPH else meta.vapor
        sigma = self.interfaces[0].functional.thermo.sigma[0]
        eps = self.interfaces[0].functional.thermo.eps_div_kb[0] * KB
        temperature = self.interfaces[0].bulk.temperature
        fac = 1.0 if self.phase == self.interfaces[0].functional.thermo.LIQPH else -1.0
        n_if = len(self.interfaces)
        n = min(n_points, n_if)
        dP = np.full(n, np.NaN)
        r_e = np.full(n, np.NaN)
        for i in range(n):
            dP[i] = self.interfaces[n_if-n+i].bulk.left_state.pressure() - self.interfaces[n_if-n+i].bulk.right_state.pressure()
            r_e[i] = self.interfaces[n_if-n+i].r_equimolar #*self.interfaces[i].functional.grid_reducing_lenght/sigma
        dP *= fac*sigma**3/eps
        interplator = interp1d(dP, r_e,
                               kind=interpolation,
                               bounds_error=False,
                               fill_value="extrapolate",
                               assume_sorted=True)
        dP_new = (left_state.pressure() - right_state.pressure())*fac*sigma**3/eps
        r_new = interplator(np.array([dP_new]))[0]
        print("r*",r_new*self.interfaces[i].functional.grid_reducing_lenght/sigma)
        plt.plot(dP, r_e)
        plt.plot([dP_new], [r_new], linestyle="None", marker="o")
        plt.show()
        return r_new

    def plot(self, base_file_name, reduced_unit=False):
        """Class for calculating surface tension along saturation curve

        Args:
            curve (PhaseDiagram): List of states to calculate surface tension
            geometry (int): PLANAR/POLAR/SPHERICAL
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            solver (dft_solver, optional): Solver for DFT
            init_profiles (ProfileInit): How to initialize profiles? Default: ProfileInit.TANH
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """
        sigma = self.interfaces[0].functional.thermo.sigma[0]
        n_if = len(self.interfaces)
        r_t = np.full(n_if, np.NaN)
        r_e = np.full(n_if, np.NaN)
        mu0 = np.full(n_if, np.NaN)
        gamma_t = np.full(n_if, np.NaN)
        gamma_e = np.full(n_if, np.NaN)
        dP = np.full(n_if, np.NaN)
        n_E = np.full(n_if, np.NaN)
        W_formation = np.full(n_if, np.NaN)
        fac = 1.0 if self.phase == self.interfaces[0].functional.thermo.LIQPH else -1.0
        for i in range(n_if):
            if self.interfaces[i].converged:
                gamma_t[i], r_t[i], delta = self.interfaces[i].surface_of_tension(reduced_unit)
                mu0[i] = self.interfaces[i].bulk.real_mu[0]
                dP[i] = fac*(self.interfaces[i].bulk.left_state.pressure() - self.interfaces[i].bulk.right_state.pressure())
                n_E[i] = self.interfaces[i].n_excess()
                W_formation[i] = self.interfaces[i].work_of_formation()
                gamma_e[i] = self.interfaces[i].surface_tension(reduced_unit)
                r_e[i] = self.interfaces[i].r_equimolar*self.interfaces[i].functional.grid_reducing_lenght/sigma
            else:
                print("Not converged: : ",i)

        xlabel = r"$\Delta P^*$" if reduced_unit else r"$\Delta P$ (MPa)"
        eps = self.interfaces[0].functional.thermo.eps_div_kb[0] * KB
        temperature = self.interfaces[0].bulk.temperature
        dP *= sigma**3/eps if reduced_unit else 1.0e-6

        plt.figure()
        gamma_t *= 1.0 if reduced_unit else 1e3
        gamma_e *= 1.0 if reduced_unit else 1e3
        plt.plot(dP, gamma_t, lw=2, color="b",
                 label=r"$\gamma_{\rm{T}}^*$" if reduced_unit else r"$\gamma_{\rm{T}}$")
        plt.plot(dP, gamma_e, lw=2, color="g",
                 label=r"$\gamma_{\rm{e}}^*$" if reduced_unit else r"$\gamma_{\rm{e}}$")
        plt.xlabel(xlabel)
        plt.ylabel(r"$\gamma^*$" if reduced_unit else r"$\gamma$ (mN/m)")
        leg = plt.legend(loc="best", numpoints=1, frameon=False)
        plt.tight_layout()
        plt.savefig(base_file_name + "_gamma.pdf")

        plt.figure()
        r_t *= 1.0 if reduced_unit else 1.0e10*sigma
        r_e *= 1.0 if reduced_unit else 1.0e10*sigma
        plt.plot(dP, r_t, lw=2, color="b",
                 label=r"$r_{\rm{T}}^*$" if reduced_unit else r"$r_{\rm{T}}$")
        plt.plot(dP, r_e, lw=2, color="g",
                 label=r"$r_{\rm{e}}^*$" if reduced_unit else r"$r_{\rm{e}}$")
        plt.xlabel(xlabel)
        plt.ylabel(r"$r^*$" if reduced_unit else r"$r$ (Å)")
        leg = plt.legend(loc="best", numpoints=1, frameon=False)
        plt.tight_layout()
        plt.savefig(base_file_name + "_radius.pdf")

        plt.figure()
        plt.plot(dP, n_E, lw=2, color="k")
        plt.xlabel(xlabel)
        plt.ylabel(r"$n_{\rm{E}}$")
        plt.tight_layout()
        plt.savefig(base_file_name + "_n_E.pdf")

        plt.figure()
        W_formation *= 1.0 if reduced_unit else NA*KB*temperature
        plt.plot(dP, W_formation, lw=2, color="k")
        plt.xlabel(xlabel)
        plt.ylabel(r"$\beta W_{\rm{F}}$" if reduced_unit else r"$W_{\rm{F}}$ (J/mol)")
        plt.tight_layout()
        plt.savefig(base_file_name + "_W_F.pdf")

        plt.show()

        # Save to file
        if reduced_unit:
            header = "# mu*, DP*, r_t*, r_e*, gamma_t*, gamma_e*, betaW, n_E"
        else:
            header = "# mu (J/mol), DP (MPa), r_t (Å), r_e (Å), gamma_t (mN/m), gamma_e (mN/m), W (J/mol), n_E"
        np.savetxt(base_file_name+".dat",
                   np.c_[mu0, dP, r_t, r_e, gamma_t, gamma_e, W_formation, n_E],
                   header=header)

class plot_animator():

    def __init__(self,
                 interfaces,
                 z_max,
                 prop,
                 reduced_property,
                 plot_equimolar_surface,
                 plot_bulk,
                 include_legend,
                 is_liquid_right,
                 y_lim=None,
                 **kw):
        self.interfaces = interfaces
        self.fig, self.ax = plt.subplots()
        self.line = None
        self.z_max = z_max
        self.is_liquid_right = is_liquid_right
        self.prop = prop
        self.reduced_property = reduced_property
        self.plot_equimolar_surface = plot_equimolar_surface
        self.plot_bulk = plot_bulk
        self.z_b = None
        self.include_legend = include_legend
        self.y_lim = y_lim

        self.ani = FuncAnimation(self.fig,
                                 self.animate2,
                                 frames=len(self.interfaces),
                                 repeat=False,
                                 interval=20,
                                 blit=False,
                                 **kw)

    def animate(self, frame):

        z, _, xlabel, _ = self.interfaces[frame].get_property_profiles(prop=Properties.GRID,
                                                                       reduced_property=self.reduced_property)
        z = z[0]
        prop_profiles, legend, ylabel, unit_scaling = self.interfaces[frame].get_property_profiles(prop=self.prop,
                                                                                                   reduced_property=self.reduced_property)

        #self.ani.event_source.stop()
        n_profiles = len(prop_profiles)
        for i in range(n_profiles):
            self.line[i].set_data(z, prop_profiles[i])

        if self.plot_bulk:
            prop_b = self.interfaces[frame].bulk.get_property(self.prop, reduced_property=self.reduced_property)
            for i in range(n_profiles):
                p_b = prop_b[i,:] if self.prop==Properties.RHO else prop_b
                self.line[i+n_profiles].set_data(self.z_b, p_b*unit_scaling)

        self.ax.relim()
        self.ax.autoscale_view()
        if self.plot_equimolar_surface:
            # Plot equimolar dividing surface
            yl = self.ax.get_ylim()
            r_equimolar = self.len_fac*self.interfaces[frame].r_equimolar
            self.line[-1].set_data([r_equimolar, r_equimolar], yl)

        if self.z_max is not None:
            self.ax.set_xlim(0, self.z_max)

        return self.line

    def animate2(self, frame):

        z, _, xlabel, _ = self.interfaces[frame].get_property_profiles(prop=Properties.GRID,
                                                                       reduced_property=self.reduced_property)
        z = z[0]
        prop_profiles, legend, ylabel, unit_scaling = self.interfaces[frame].get_property_profiles(prop=self.prop,
                                                                                                   reduced_property=self.reduced_property)
        self.z_b = np.array([z[0], z[-1]])
        self.len_fac = z[-1]/self.interfaces[0].grid.z[-1]

        self.ax.clear()
        #self.ani.event_source.stop()
        n_profiles = len(prop_profiles)
        for i in range(n_profiles):
            line, = self.ax.plot(z, prop_profiles[i],
                                 lw=2, color=LCOLORS[i], label=legend[i])

        if self.plot_bulk:
            prop_b = self.interfaces[frame].bulk.get_property(self.prop, reduced_property=self.reduced_property)
            label = "Bulk"
            for i in range(n_profiles):
                p_b = prop_b[i,:] if self.prop==Properties.RHO else prop_b
                line, = self.ax.plot(self.z_b, p_b*unit_scaling, color=LCOLORS[i], marker="o", linestyle="None", label=label)
                label = None

        if self.plot_equimolar_surface:
            # Plot equimolar dividing surface
            if self.y_lim is not None:
                yl = self.y_lim
            else:
                yl = self.ax.get_ylim()
            r_equimolar = self.len_fac*self.interfaces[frame].r_equimolar
            line, = self.ax.plot([r_equimolar, r_equimolar],
                                  yl,
                                  lw=1, color="k",
                                  linestyle="--",
                                  label="Eq. mol. surf.")

        if self.include_legend:
            pos = 4 if self.is_liquid_right else 1
            self.ax.legend(loc=pos, numpoints=1, frameon=False)

        if self.z_max is not None:
            self.z_max = self.len_fac*self.interfaces[0].grid.domain_size

        self.ax.set_xlim(0, self.z_max)
        if self.y_lim is not None:
            self.ax.set_ylim(self.y_lim)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        return

    def init_plot(self):
        z, _, xlabel, _ = self.interfaces[0].get_property_profiles(prop=Properties.GRID,
                                                                   reduced_property=self.reduced_property)
        z = z[0]
        prop_profiles, legend, ylabel, unit_scaling = self.interfaces[0].get_property_profiles(prop=self.prop,
                                                                                               reduced_property=self.reduced_property)

        n_prof = len(prop_profiles)
        self.line = []
        for i in range(n_prof):
#            line, = self.ax.plot(z, prop_profiles[i],
#                                 lw=2, color=LCOLORS[i], label=legend[i])
            line, = self.ax.plot([], [],
                                 lw=2, color=LCOLORS[i], label=legend[i])
            self.line.append(line)

        if self.plot_bulk:
            self.z_b = np.array([z[0], z[-1]])
            prop_b = self.interfaces[0].bulk.get_property(self.prop, reduced_property=self.reduced_property)
            for i in range(n_prof):
                p_b = prop_b[i,:] if self.prop==Properties.RHO else prop_b
#                line, = self.ax.plot(self.z_b, p_b*unit_scaling, color=LCOLORS[i], marker="o", linestyle="None")
                line, = self.ax.plot([], [], color=LCOLORS[i], marker="o", linestyle="None")
                self.line.append(line)

        self.len_fac = z[-1]/self.interfaces[0].grid.z[-1]
        if self.plot_equimolar_surface:
            # Plot equimolar dividing surface
            yl = self.ax.get_ylim()
            r_equimolar = self.len_fac*self.interfaces[0].r_equimolar
            # line, = self.ax.plot([r_equimolar, r_equimolar],
            #                      yl,
            #                      lw=1, color="k",
            #                      linestyle="--",
            #                      label="Eq. mol. surf.")
            line, = self.ax.plot([],
                                 [],
                                 lw=1, color="k",
                                 linestyle="--",
                                 label="Eq. mol. surf.")

            self.line.append(line)

        if self.include_legend:
            self.ax.legend(loc="best", numpoints=1, frameon=False)

        if self.z_max is not None:
            self.z_max = self.len_fac*self.interfaces[0].grid.domain_size
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        return self.line

    def save(self, base_file_name="animation"):
        self.ani.save(base_file_name + ".gif", writer='pillow') #'imagemagick', 'pillow'

if __name__ == "__main__":
    pass
