#!/usr/bin/env python3
import numpy as np
from cdft import cdft1D
import matplotlib.pyplot as plt

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
        self.domain_mask = np.empty(self.cDFT.N, dtype=bool)
        self.domain_mask[:] = False
        self.domain_mask[self.NiWall:self.cDFT.end] = True
        self.wall_mask = np.full(self.cDFT.N, False, dtype=bool)
        self.wall_mask[self.NiWall] = True

        # Set radius for plotting
        self.r = np.linspace(-self.NiWall*self.cDFT.dr, (self.cDFT.end-1)*self.cDFT.dr, self.cDFT.N)

        # Density profile
        self.density = np.zeros(self.cDFT.N)
        self.density[self.domain_mask] = self.cDFT.bulk_density
        self.old_density = np.zeros(self.cDFT.N)
        self.old_density = self.density
        print(self.cDFT.N)
        print("self.domain_mask", self.domain_mask)
        self.new_density = np.zeros(self.cDFT.N)
        self.mod_density = np.zeros(self.cDFT.N)

        # Set state of solver
        self.converged = False

        # Set Picard paramater for relaxed successive substitution
        self.new_fraction = new_fraction

        # Error norm (np.inf: Max norm, None: 2 norm, ....)
        self. norm = np.inf

    def picard_update(self):
        """
        Updates density profile using Picard procedures.

        Returns:
            error (float): Deviation between current and previous density profile
            have_failed (bool): Fail indicator
        """

        # Calculate weighted densities
        self.mod_density = self.density
        self.mod_density[self.wall_mask] *= 0.5  # todo find explanation for 0.5 trick
        self.cDFT.weights.convolutions(self.cDFT.weighted_densities, self.mod_density)

        # Calculate one-body direct correlation function
        self.cDFT.weights.correlation_convolution(self.cDFT.differentials)
        corrHS = self.cDFT.differentials.corr

        # Calculate new density profile using the variations of the functional
        self.new_density[self.domain_mask] = self.cDFT.bulk_density * \
                                             np.exp(corrHS[self.domain_mask] + self.cDFT.beta *
                                                    (self.cDFT.excess_mu - self.cDFT.Vext[self.domain_mask]))

        print("new-density",self.new_density[self.domain_mask])
        # Set old density profile
        self.old_density = self.density

        # Check for valid floats
        if np.any(np.isnan(self.new_density)) or np.any(np.isinf(self.new_density)):
            have_failed = True
        else:
            have_failed = False

        # Picard update
        self.density = (1.0 - self.new_fraction) * self.density + \
                                       self.new_fraction * self.new_density

        print("density", self.density)
        print("old-density", self.old_density)
        print("diff density", self.density - self.old_density)
        # Calculate deviation between new and old density profiles
        error = np.linalg.norm(self.density - self.old_density, ord=self.norm)
        print("error",error)
        return error, have_failed

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
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel("$z$")
            ax.set_ylabel(r"$\rho^*$")
            d_plot, = ax.plot(self.r[self.domain_mask], self.density[self.domain_mask], lw=2, color="k")
            d_plot_old, = ax.plot(self.r[self.domain_mask], self.old_density[self.domain_mask],
                                  lw=2, color="k", ls="--")
        error = 1.0
        for iteration in range(maximum_iterations):
            error, have_failed = self.picard_update()
            if have_failed:
                break
            if error < tolerance:
                self.converged = True
                break

            if iteration % print_frequency == 0:
                print("{} complete. Deviation: {}\n".format(iteration, error))
                if plot_profile:
                    d_plot.set_ydata(self.density[self.domain_mask])
                    d_plot_old.set_ydata(self.old_density[self.domain_mask])
                    plt.draw()

        if not self.converged:
            print("Solver did not converge. Deviation at exit {}\n".format(error))

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

    def plot_equilibrium_density_profile(self):
        """
        Plots the equilibrium density profile.
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
    bulk_density = 0.38 #density_from_packing_fraction(eta=0.2)
    dft = cdft1D(bulk_density=bulk_density, domain_length=1.0, wall="HardWall", grid_dr=0.5)
    solver = picard_geometry_solver(cDFT=dft)
    solver.density[2] *= 0.5
    solver.density[-1] = 0.38
    solver.density[-2] = 0.38
    #solver.density[6] = 0.38

    dft.weights.w2 = np.array([0.5645049299419159, 2.138028333693054, 0.5645049299419159])
    dft.weights.w3 = np.array([0.0, 0.5345070834232635, 0.0])
    dft.weights.w2vec = np.array([- 0.5645049299419159, 0.0, 0.5645049299419159])

    dft.weights.convolutions(dft.weighted_densities, solver.density)
    print(solver.density)
    print("w2",dft.weights.w2)
    print("w3", dft.weights.w3)
    print("w2v", dft.weights.w2vec)
    dft.weighted_densities.n2[:2] = 0.0
    dft.weighted_densities.n3[:2] = 0.0
    dft.weighted_densities.n2v[:2] = 0.0
    dft.weighted_densities.n0[:2] = 0.0
    dft.weighted_densities.n1[:2] = 0.0
    dft.weighted_densities.n1v[:2] = 0.0
    print("n2",dft.weighted_densities.n2)
    print("n3",dft.weighted_densities.n3)
    print("n2v",dft.weighted_densities.n2v)

    dft.weighted_densities.print()

    dft.functional.differentials(dft.weighted_densities, dft.differentials)
    dft.differentials.print()
    dft.weights.correlation_convolution(dft.differentials)
    print("c2", dft.differentials.d2eff_conv)
    print("c3", dft.differentials.d3_conv)
    print("\nc2v", dft.differentials.d2veff_conv)
    print("\ncorr: ",dft.differentials.corr)

    solver.picard_update()
    #picard_geometry_solver.minimise()

