import itertools
from surfpack.saft import SAFT
from surfpack.saft_hardsphere import SAFT_WhiteBear
from thermopack.pcsaft import pcsaft
from scipy.constants import Avogadro
import numpy as np
from surfpack import Profile
from surfpack.WeightFunction import NormTheta, NormTheta_diff, Delta, Delta_diff, Heaviside, Heaviside_diff, LocalDensity
import copy
from surfpack.Convolver import convolve_ad
from collections.abc import Iterable

class PC_SAFT(SAFT):
    def __init__(self, comps, hs_model=SAFT_WhiteBear, parameter_ref='default'):
        """Constructor
        Initialises a PC-SAFT model.

        Args:
            comps (str) : Comma separated component identifiers, following thermopack convention.
            hs_model (SAFT_HardSphere) : Model to use for the hard-sphere contribution. Default is SAFT_WhiteBear.
            parameter_ref (str) : Reference for parameter set to use (see ThermoPack).
        """
        super().__init__(comps, pcsaft, hs_model, parameter_ref=parameter_ref)

    def __repr__(self):
        """Internal
        Generates a unique string for this model, containing information about the parameters and hard-sphere model.
        """
        ostr = f'pc-saft model for {self._comps},\n' + super().__repr__()
        return ostr

    def get_caching_id(self):
        ostr = f'PC-SAFT {self._comps}\n' + super().get_caching_id()
        return ostr

    def get_weights(self, T, dwdT=False):
        """Weights
        Get all the weights used for weighted densities in a 2D array, indexed as weight[<wt idx>][<comp idx>].

        Args:
            T (float) : Temperature [K], used to get hard sphere diameters
            dwdT (bool) : Compute temperature differentials

        Return:
            2D array [Analytical] : Weight functions, indexed as weight[<wt idx>][<comp idx>]
        """
        if T in self.computed_weights.keys():
            return copy.deepcopy(self.computed_weights[T])

        d, dd_dT = self.eos.hard_sphere_diameters(T)
        d *= 1e10

        if dwdT is False:
            weights = super().get_weights(T)
            weights.extend(self.get_association_weights(T))
            weights.extend(self.get_chain_weights(T))
            self.computed_weights[T] = copy.deepcopy(weights)
            return weights

        dwdT = super().get_weights(T, dwdT=True)
        dwdT.extend(self.get_association_weights(T, dwdT=True))
        dwdT.extend(self.get_chain_weights(T, dwdT=True))
        return dwdT

    def get_association_weights(self, T, dwdT=False):
        """Weights
        Get the weight functions for the association weighted densities, indexed as wts[<wt_idx>][<comp_idx>], equivalent
        to the component weights from FMT.

        Args:
            T (float) : Temperature [K]
            dwdT (bool) : Compute derivative

        Returns:
            list[list[Analytical]] : The weight functions.
        """
        if dwdT is True:
            dwdT_assoc = [[0 for _ in range(self.ncomps)] for _ in range(6 * self.ncomps)]
            dwdT_fmt = self.get_fmt_weights(T, dwdT=True)
            for ci in range(self.ncomps):
                for wi in range(6):
                    dwdT_assoc[ci * 6 + wi][ci] = dwdT_fmt[wi][ci]
            return dwdT_assoc

        w_assoc = [[0 for _ in range(self.ncomps)] for _ in range(6 * self.ncomps)]
        w_fmt = self.get_fmt_weights(T)
        for ci in range(self.ncomps):
            for wi in range(6):
                w_assoc[ci * 6 + wi][ci] = w_fmt[wi][ci]

        return w_assoc

    def get_chain_weights(self, T, dwdT=False):
        """Weights
        Get the weight functions for the chain contribution, indexed as wts[<wt_idx>][<comp_idx>]. The weights
        wts[0] correspond to the local density, wts[1] are eq. Eq. 53 in Sauer & Gross, 10.1021/acs.iecr.6b04551 (lambda)
        and wts[1] are Eq. 54 in the same paper.

        Args:
            T (float) : Temperature [K]
            dwdT (bool) : Compute derivative instead.
        Returns:
            list[list[Analytical]] : The weight functions, shape (3 * ncomps, ncomps)
        """
        d, d_dT = self.eos.hard_sphere_diameters(T)
        d *= 1e10
        d_dT *= 1e10

        if dwdT is True:
            dw_chain_dT = [[0 for _ in range(self.ncomps)] for _ in range(3)]
            for i in range(self.ncomps):
                dw_chain_dT[0][i] = 0
                dw_chain_dT[1][i] = ((Delta_diff(d[i]) * d[i] - 2 * Delta(d[i])) / (4 * np.pi * d[i]**3)) * d_dT[i]
                dw_chain_dT[2][i] = (3 / 4) * ((Heaviside_diff(d[i]) * d[i] - 3 * Heaviside(d[i])) / d[i]**4) * d_dT[i]
            return dw_chain_dT

        w_chain = [[0 for _ in range(self.ncomps)] for _ in range(3 * self.ncomps)]
        for i in range(self.ncomps):
            w_chain[i][i] = LocalDensity()
            w_chain[self.ncomps + i][i] =  Delta(d[i]) / (4 * np.pi * d[i]**2) # lambda weight - Eq. 53 in Sauer & Gross, 10.1021/acs.iecr.6b04551
            w_chain[2 * self.ncomps + i][i] = Heaviside(d[i]) / (4 * np.pi * d[i]**3 / 3) # \bar{\rho}^{hc} - Eq. 54
        return w_chain

    def get_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False):
        """Weighted density
        Compute all neccessary weighted densities for this model.

        Args:
            rho (list[Profile] or list[float]) : The particle density of each species
            T (float) : Temperature [K]
            bulk (bool) : Default False. Set to True if `rho` is `list[float]`
            dndrho (bool) : Compute derivative (only for `bulk=True`)
            dndT (bool) : Compute derivative

        Returns:
            list[Profile] or list[float] : Weighted densities
            Optional 2d array (float) : dndrho, derivatives of weighted densities, indexed as dndrho[<wt idx>][<comp idx>]
        """
        if dndrho is True: # Only if bulk is True
            n_fmt_comp, dndrho_fmt_comp = self.hs_model.get_component_weighted_densities(rho, T, bulk=bulk, dndrho=True)
            n_hs, dndrho_hs = self.hs_model.get_weighted_densities(rho, T, bulk=bulk, dndrho=dndrho, n_comp=n_fmt_comp)
            n_disp, dndrho_disp = self.get_dispersion_weighted_density(rho, T, bulk=bulk, dndrho=dndrho)
            n_assoc, dndrho_assoc = [*itertools.chain.from_iterable(n_fmt_comp)], dndrho_fmt_comp
            n_chain, dndrho_chain = self.get_chain_weighted_densities(rho, T, bulk=bulk, dndrho=dndrho)
            dndrho = [*dndrho_hs, *dndrho_disp, *dndrho_assoc, *dndrho_chain]
            n = [*n_hs, *n_disp, *n_assoc, *n_chain]
            return n, dndrho
        if dndT is True:
            n_super, dndT_super = super().get_weighted_densities(rho, T, bulk=bulk, dndT=dndT)
            n_assoc, dndT_assoc = self.get_assoc_weighted_densities(rho, T, bulk=bulk, dndT=dndT)
            return n_super + n_assoc, dndT_super + dndT_assoc

        n_fmt_comp = self.hs_model.get_component_weighted_densities(rho, T, bulk=bulk)
        n_hs = self.hs_model.get_weighted_densities(rho, T, bulk=bulk, n_comp=n_fmt_comp)
        n_disp = self.get_dispersion_weighted_density(rho, T, bulk=bulk)
        n_assoc = [*itertools.chain.from_iterable(n_fmt_comp)]
        n_chain = self.get_chain_weighted_densities(rho, T, bulk=bulk)
        n = [*n_hs, *n_disp, *n_assoc, *n_chain]
        return n

    def get_assoc_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False):
        """Weighted density
        Compute the component weighted densities

        Args:
            rho (list[Profile]) : Density profiles [particles / Å^3] indexed as rho_arr[<comp idx>][<grid idx>]
            T (float) : Temperature [K], used to get radii and weights (in case of temp. dependent radii)
            bulk (bool) : If true, take a list[float] for rho_arr, and return a list[float]
            dndrho (bool) : If true, also return derivative (only for bulk)
            dndT (bool) : If true, also return derivative
        Returns:
            list[Profile] : weighted densities [particles / Å^3], indexed as wts[<comp idx>][<grid idx>]
            Optional 2d array (float) : dndrho, derivatives of weighted densities, indexed as dndrho[<wt idx>][<comp idx>]
        """
        if dndrho is True:
            n_assoc, dndrho_assoc = self.hs_model.get_component_weighted_densities(rho, T, bulk=bulk, dndrho=True)
            return [*itertools.chain.from_iterable(n_assoc)], dndrho_assoc
        elif dndT is True:
            n_assoc, dndT_assoc = self.hs_model.get_component_weighted_densities(rho, T, bulk=bulk, dndT=True)
            return [*itertools.chain.from_iterable(n_assoc)], dndT_assoc
        n_assoc = self.hs_model.get_component_weighted_densities(rho, T, bulk=bulk)
        return [*itertools.chain.from_iterable(n_assoc)]

    def get_chain_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False):
        """Weighted density
        Compute the weighted densities for the chain contribution

        Args:
            rho (list[Profile]) : Density profiles [particles / Å^3] indexed as rho_arr[<comp idx>][<grid idx>]
            T (float) : Temperature [K], used to get radii and weights (in case of temp. dependent radii)
            bulk (bool) : If true, take a list[float] for rho_arr, and return a list[float]
            dndrho (bool) : If true, also return derivative (only for bulk)
            dndT (bool) : If true, also return derivative
        Returns:
            list[Profile] : weighted densities [particles / Å^3], indexed as wts[<comp idx>][<grid idx>]
            Optional 2d array (float) : dndrho, derivatives of weighted densities, indexed as dndrho[<wt idx>][<comp idx>]
        """
        w_chain = self.get_chain_weights(T, dwdT=False)

        if bulk is True:
            n_chain = np.zeros(3 * self.ncomps)
            for i, rho_i in enumerate(rho):
                n_chain[i] = rho_i * w_chain[i][i].real_integral()
                n_chain[self.ncomps + i] = rho_i * w_chain[self.ncomps + i][i].real_integral()
                n_chain[2 * self.ncomps + i] = rho_i * w_chain[2 * self.ncomps + i][i].real_integral()
        else:
            n_chain = Profile.zeros(rho[0].grid, nprofiles=3 * self.ncomps)
            for i, rho_i in enumerate(rho):
                n_chain[i] = Profile(convolve_ad(w_chain[i][i], rho_i), rho_i.grid)
                n_chain[self.ncomps + i] = Profile(convolve_ad(w_chain[self.ncomps + i][i], rho_i), rho_i.grid) # lambda weighted density - Eq. 53 in Sauer & Gross, 10.1021/acs.iecr.6b04551
                n_chain[2 * self.ncomps + i] = Profile(convolve_ad(w_chain[2 * self.ncomps + i][i], rho_i), rho_i.grid) # \bar{\rho}^{hc} - Eq. 54

        if dndrho is True:
            if bulk is False:
                raise NotImplementedError('dndrho only implemented for bulk!')

            dndrho = np.zeros((3 * self.ncomps, self.ncomps))
            for i in range(self.ncomps):
                dndrho[i, i] = w_chain[i][i].real_integral()
                dndrho[self.ncomps + i][i] = w_chain[self.ncomps + i][i].real_integral()
                dndrho[2 * self.ncomps + i][i] = w_chain[2 * self.ncomps + i][i].real_integral()

            return n_chain, dndrho

        if dndT is True:
            dndT = Profile.zeros(rho[0].grid, nprofiles=3 * self.ncomps)
            dwdT = self.get_chain_weights(T, dwdT=True)
            for i, rho_i in enumerate(rho):
                dndT[i] = Profile.zeros(rho_i.grid)
                dndT[self.ncomps + i] = Profile(convolve_ad(dwdT[2 * i][i], rho_i), rho_i.grid)
                dndT[2 * self.ncomps + i] = Profile(convolve_ad(dwdT[2 * i + 1][i], rho_i), rho_i.grid)
            return n_chain, dndT

        return n_chain

    def association_helmholtz_energy_density(self, rho, T, bulk=False, dphidn=False, dphidT=False, dphidrho=False, n_assoc=None):
        """Helmholtz contribution
        Compute the association contribution to the reduced Helmholtz energy density.

        Args:
            rho (list[Profile] or list[float]) : Particle density for each species
            T (float) : Temperature [K]
            bulk (bool) : Default False. Set to True if `rho` is `list[float]`
            dphidn (bool) : Compute derivative
            dphidT (bool) : Compute derivative
            dphidrho (bool) : Compute derivative
            n_assoc (list[Profile], optional) : Pre-computed weighted densities.

        Returns:
            Profile or float : The (reduced) association helmholtz energy density [-]
        """
        if self.__association_active__ is False:
            if (dphidn is True) or (dphidT is True):
                return (0, np.zeros(6 * self.ncomps)) if (bulk is True) else (Profile.zeros_like(rho), Profile.zeros(rho[0].grid, 6 * self.ncomps))
            if dphidrho is True:
                return (0, np.zeros(self.ncomps)) if (bulk is True) else (Profile.zeros_like(rho), Profile.zeros(rho[0].grid, self.ncomps))
            return 0 if (bulk is True) else np.zeros(len(rho[0]))

        if dphidrho is True:
            phi, dphidn = self.association_helmholtz_energy_density(rho, T, bulk=bulk, dphidn=True)
            w = self.get_association_weights(T)
            if bulk is True:
                dphidrho = np.zeros(self.ncomps)
                for i in range(self.ncomps):
                    for j in range(6):
                        if w[j * self.ncomps + i][i] == 0:
                            continue
                        dphidrho[i] += dphidn[j * self.ncomps + i] * w[j * self.ncomps + i][i].real_integral()
            else:
                dphidrho = Profile.zeros_like(rho)
                for i in range(self.ncomps):
                    for j in range(6):
                        dphidrho[i] += convolve_ad(w[j * self.ncomps + i][i], dphidn[j * self.ncomps + i])

            return phi, dphidrho

        if dphidT is True:
            n_assoc, dndT = self.get_assoc_weighted_densities(rho, T, bulk=bulk, dndT=True)
        elif n_assoc is None:
            n_assoc = self.get_assoc_weighted_densities(rho, T, bulk=bulk)

        wt_conv_f = self.hs_model.get_weighted_density_conversion_factors()

        if bulk is True:  # When computing chemical potential bulk is True.
            n_assoc = [[n_assoc[6 * ci + wi] for wi in range(6)] for ci in range(self.ncomps)]
            n_alpha = np.array(self.hs_model.weighted_densities_to_SI(n_assoc))
            if dphidn is True:
                a, a_n = self.eos.association_energy_density(T, n_alpha, phi=True, phi_n=True)
            elif dphidT is True:
                a, a_T, a_n = self.eos.association_energy_density(T, n_alpha, phi=True, phi_t=True, phi_n=True)
            else:
                a, = self.eos.association_energy_density(T, n_alpha, phi=True)

            phi_assoc = a * Avogadro / 1e30
            if dphidn is True:
                dphidn_assoc = [[None for _ in range(6)] for _ in range(self.ncomps)]
                for ci in range(self.ncomps):
                    for wi in range(6):
                        dphidn_assoc[ci][wi] = a_n[ci][wi] * (Avogadro / 1e30) * wt_conv_f[wi]

                dphidn_assoc = [*itertools.chain.from_iterable(dphidn_assoc)]
                return phi_assoc, dphidn_assoc

            if dphidT is True:
                dphidT_assoc = a_T
                return phi_assoc, dphidT_assoc

            return phi_assoc

        ### Bulk is False ###
        n_assoc = [[n_assoc[6 * ci + wi] for wi in range(6)] for ci in range(self.ncomps)]
        phi_assoc = Profile.zeros(n_assoc[0][0].grid)  # Returns Profile, with grid taken from n
        if dphidn is True:
            dphidn_assoc = [Profile.zeros_like(n_assoc[ci]) for ci in range(self.ncomps)]
        elif dphidT is True:
            dphidT_assoc = Profile.zeros(n_assoc[0][0].grid)

        xk_init = [0.2 for _ in range(self.eos.get_n_assoc_sites())] # Initial guess for fraction of non-bonded association sites (updated while iterating through density profile)
        for ri in range(len(phi_assoc)):  # Iterate over the grid positions
            n_alpha = np.array([[n_assoc[ci][wi][ri] for wi in range(6)] for ci in range(self.ncomps)])
            n_alpha = self.hs_model.weighted_densities_to_SI(n_alpha)
            if abs(sum(sum(n_alpha))) < 1.1e-12:  # Ensure that thermopack does not return NaN
                a, a_n = 0, np.zeros((self.ncomps, 6))
            else:
                if dphidn is True:
                    a, a_n, xk_init = self.eos.association_energy_density(T, n_alpha, phi=True, phi_n=True, Xk=xk_init)  # Gives nan if sum(n) == 0
                elif dphidT is True:
                    a, a_T, a_n, xk_init = self.eos.association_energy_density(T, n_alpha, phi=True, phi_t=True, phi_n=True, Xk=xk_init)
                else:
                    a, xk_init = self.eos.association_energy_density(T, n_alpha, phi=True, Xk=xk_init)

            phi_assoc[ri] = a * Avogadro / 1e30
            if dphidn is True:
                for ci in range(self.ncomps):
                    for wi in range(6):
                        dphidn_assoc[ci][wi][ri] = a_n[ci][wi] * (Avogadro / 1e30) * wt_conv_f[wi]
            elif dphidT is True:
                dphidT_assoc[ri] = a_T

        if dphidn is True:
            dphidn_assoc = [*itertools.chain.from_iterable(dphidn_assoc)]
            return phi_assoc, dphidn_assoc
        elif dphidT is True:
            return phi_assoc, dphidT_assoc
        return phi_assoc

    def chain_helmholtz_energy_density(self, rho, T, dphidrho=False, dphidn=False, dphidT=False, bulk=False, n_chain=None):
        """Helmholtz contribution
        Compute the chain contribution to the reduced Helmholtz energy density

        Args:
            rho (list[Profile] or list[float]) : Particle density for each species
            T (float) : Temperature [K]
            bulk (bool) : Default False. Set to True if `rho` is `list[float]`
            dphidn (bool) : Compute derivative
            dphidT (bool) : Compute derivative
            dphidrho (bool) : Compute derivative
            n_chain (list[Profile], optional) : Pre-computed weighted densities.

        Returns:
            Profile or float : The (reduced) chain helmholtz energy density [-]
        """
        if (self.__chain_active__ is False) or all(self.ms == 1):
            if (dphidn is True) or (dphidT is True):
                return (0, np.zeros(3 * self.ncomps)) if (bulk is True) else (Profile.zeros_like(rho), Profile.zeros(rho[0].grid, 3 * self.ncomps))
            if dphidrho is True:
                return (0, np.zeros(self.ncomps)) if (bulk is True) else (Profile.zeros_like(rho), Profile.zeros(rho[0].grid, self.ncomps))
            return 0 if (bulk is True) else np.zeros(len(rho[0]))

        if dphidrho is True:
            phi, dphidn = self.chain_helmholtz_energy_density(rho, T, dphidn=True, bulk=bulk)
            w = self.get_chain_weights(T)
            if bulk is True:
                dphidrho = np.zeros(self.ncomps)
                for i in range(self.ncomps):
                    for j in range(3):
                        dphidrho[i] += dphidn[j * self.ncomps + i] * w[j * self.ncomps + i][i].real_integral()
            else:
                dphidrho = Profile.zeros_like(rho)
                for i in range(self.ncomps):
                    for j in range(3):
                        dphidrho[i] += convolve_ad(w[j * self.ncomps + i][i], dphidn[j * self.ncomps + i])

            return phi, dphidrho


        if dphidT is True:
            n_chain, dndT = self.get_chain_weighted_densities(rho, T, bulk=bulk, dndT=True)
            dldadT = dndT[:self.ncomps]
            dlambdT = dndT[self.ncomps: 2 * self.ncomps]
            dhcdT = dndT[2 * self.ncomps:]
        elif n_chain is None:
            n_chain = self.get_chain_weighted_densities(rho, T, bulk=bulk)

        n_lda = n_chain[:self.ncomps] # Local density
        n_lamb = n_chain[self.ncomps : 2 * self.ncomps] # Lambda - Eq. 53 in Sauer & Gross, 10.1021/acs.iecr.6b04551
        n_hc = n_chain[2 * self.ncomps:] # \bar{\rho}^{hc} - Eq. 54

        if bulk is True:
            phi = 0
            n_hc = np.array(n_hc) * 1e30 / Avogadro  # Converting to SI (mol / m^3)
            lng = np.zeros(self.ncomps)

            if dphidn is True:
                dlngdn = np.zeros((self.ncomps, self.ncomps))
                for i in range(self.ncomps):
                    lng[i], dlngdn[i, :] = self.eos.lng_ii(T, 1, n_hc, i + 1, lng_n=True) # Note: (dlng / drho) = V * (dlng / dn)
                dlngdn *= 1e30 / Avogadro # Converting to PKÅ (Å^3 / particle)
            else:
                for i in range(self.ncomps):
                    lng[i], = self.eos.lng_ii(T, 1, n_hc, i + 1)

            for i in range(self.ncomps):
                if n_lda[i] < 1e-12: # Handling divide by zero and log(0) (n = 0 => phi = 0)
                    continue
                phi += (self.ms[i] - 1) * n_lda[i] * (np.log(n_lda[i]) - lng[i] - np.log(n_lamb[i]))

            if dphidn is True:
                dphidn = np.zeros(3 * self.ncomps)

                for i in range(self.ncomps):
                    if n_lda[i] < 1e-12: # Handling divide by zero and log(0)
                        dphidn[i] = 0
                        dphidn[self.ncomps + i] = - (self.ms[i] - 1)
                        dphidn[2 * self.ncomps + i] = 0
                        continue
                    dphidn[i] = (self.ms[i] - 1) * (np.log(n_lda[i]) + 1 - lng[i] - np.log(n_lamb[i]))
                    dphidn[self.ncomps + i] = - (self.ms[i] - 1) * n_lda[i] / n_lamb[i]
                    for j in range(self.ncomps):
                        dphidn[2 * self.ncomps + i] += - (self.ms[j] - 1) * n_lda[j] * dlngdn[j][i]
                return phi, dphidn

            return phi

        # bulk is False
        phi = Profile.zeros(rho[0].grid)
        dphidn_chain = Profile.zeros(rho[0].grid, len(n_chain))
        for ri in range(len(rho[0])):
            n_lda_r = [n_lda[ci][ri] for ci in range(self.ncomps)]
            n_hc_r = [n_hc[ci][ri] for ci in range(self.ncomps)]
            n_lamb_r = [n_lamb[ci][ri] for ci in range(self.ncomps)]
            n_chain_r = [*n_lda_r, *n_lamb_r, *n_hc_r]
            if dphidn is True:
                phi[ri], dphidn_r = self.chain_helmholtz_energy_density(rho, T, dphidn=True, bulk=True, n_chain=n_chain_r)
                for ni in range(len(n_chain)):
                    dphidn_chain[ni][ri] = dphidn_r[ni]
            else:
                phi[ri] = self.chain_helmholtz_energy_density(rho, T, bulk=True, n_chain=n_chain_r)

        if dphidn is True:
            return phi, dphidn_chain

        return phi

    def reduced_helmholtz_energy_density(self, rho, T, dphidn=False, bulk=False, asarray=False, dphidT=False, dphidrho=False):
        """Profile Property
        Compute the the reduced Helmholtz energy density

        Args:
            rho (list[Profile] or list[float]) : Particle density for each species
            T (float) : Temperature [K]
            bulk (bool) : Default False. Set to True if `rho` is `list[float]`
            dphidn (bool) : Compute derivative
            dphidT (bool) : Compute derivative
            dphidrho (bool) : Compute derivative
            n_chain (list[Profile], optional) : Pre-computed weighted densities.

        Returns:
            Profile or float : The (reduced) chain helmholtz energy density [-]
        """
        if dphidrho is True:
            phi, dphidn = self.reduced_helmholtz_energy_density(rho, T, bulk=bulk, dphidn=True)
            w = self.get_weights(T)
            if bulk is True:
                dphidrho = np.zeros(self.ncomps)
                for wi, weights in enumerate(w):
                    for ci in range(self.ncomps):
                        if w[wi][ci] == 0:
                            continue
                        dphidrho[ci] += dphidn[wi] * w[wi][ci].real_integral()
            else:
                c = self.correlation(rho, T)
                dphidrho = [-ci for ci in c]
            return phi, dphidrho

        # Note: The following block where n is computed and distributed between different variables is only an
        # optimization that lets us re-use a bunch of convolutions. The various helmholtz energy contributions can
        # be called directly, but then a lot of convolutions will be computed several times. The method
        # get_weighted_densities is designed to re-use convolutions as much as possible.
        n = self.get_weighted_densities(rho, T, bulk=bulk)

        N_fmt = 6 # Number of FMT weighted densities
        N_disp = self.ncomps # Number of dispersion weighted densities
        N_assoc = 6 * self.ncomps # Number of association weighted densities
        N_chain = 3 * self.ncomps # Number of chain weighted densities

        fmt_start = 0
        disp_start = fmt_end = fmt_start + N_fmt
        assoc_start = disp_end = disp_start + N_disp
        chain_start = assoc_end = assoc_start + N_assoc
        chain_end = chain_start + N_chain

        n_fmt = n[fmt_start : fmt_end]
        n_disp = n[disp_start : disp_end]
        n_assoc = n[assoc_start : assoc_end]
        n_chain = n[chain_start : chain_end]

        # Now that we've computed all the neccessary weighted densities, we can pass them to the Helmholtz energy
        # contributions, so that they don't need to re-compute a bunch of weighted densities.
        if dphidn is True:
            phi_hs, dphidn_hs = self.hs_model.reduced_helmholtz_energy_density(rho, T, dphidn=dphidn, bulk=bulk, n_fmt=n_fmt)
            phi_disp, dphidn_disp = self.dispersion_helmholtz_energy_density(rho, T, dphidn=dphidn, bulk=bulk, n_disp=n_disp)
            phi_assoc, dphidn_assoc = self.association_helmholtz_energy_density(rho, T, dphidn=dphidn, bulk=bulk, n_assoc=n_assoc)
            phi_chain, dphidn_chain = self.chain_helmholtz_energy_density(rho, T, dphidn=dphidn, bulk=bulk)
            return phi_hs + phi_disp + phi_assoc + phi_chain, [*dphidn_hs, *dphidn_disp, *dphidn_assoc, *dphidn_chain]

        elif dphidT is True:
            phi, dphidT_0 = super().reduced_helmholtz_energy_density(rho, T, dphidT=dphidT, bulk=bulk)
            phi_assoc, dphidT_assoc = self.association_helmholtz_energy_density(rho, T, dphidT=dphidT, bulk=bulk)
            return phi + phi_assoc, dphidT_0 + dphidT_assoc

        phi_hs = self.hs_model.reduced_helmholtz_energy_density(rho, T, bulk=bulk, n_fmt=n_fmt)
        phi_disp = self.dispersion_helmholtz_energy_density(rho, T, bulk=bulk, n_disp=n_disp)
        phi_assoc = self.association_helmholtz_energy_density(rho, T, bulk=bulk, n_assoc=n_assoc)
        phi_chain = self.chain_helmholtz_energy_density(rho, T, bulk=bulk)
        return phi_hs + phi_disp + phi_assoc + phi_chain

    def get_characteristic_lengths(self):
        """Utility
        Compute a characteristic length for the system.

        Returns:
            float : The sigma-parameter of the first component.
        """
        return self.eos.get_pure_params(1)[1] * 1e10