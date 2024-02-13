from surfpack.Functional import Functional
from surfpack.saft_hardsphere import SAFT_WhiteBear
from scipy.constants import Avogadro, gas_constant, Boltzmann
import numpy as np
from surfpack.profile import Profile
from surfpack.WeightFunction import NormTheta, NormTheta_diff, get_FMT_weights, get_FMT_weight_derivatives
from surfpack.Convolver import convolve_ad
import copy

class SAFT(Functional):

    def __init__(self, comps, eos, hs_model=SAFT_WhiteBear, parameter_ref='default'):
        """Internal
        This class is inherited by SAFT_VR_Mie and PC_SAFT. The only thing they do is pass the correct eos to the
        eos argument of this initialiser.

        Args:
            comps (str) : Thermopack component string
            eos (thermo) : Thermopack EoS class (not initialised object)
            hs_model (SAFT_HardSphere) : A class of the same type as SAFT_WhiteBear (inheriting from SAFT_HardSphere)
        """
        ncomps = len(comps.split(','))
        super().__init__(ncomps)

        self.__HS_MODEL__ = hs_model
        self.__EOS__ = eos

        self.eos = eos(comps, parameter_reference=parameter_ref)
        self._comps = comps
        self.ms = np.array([self.eos.get_pure_fluid_param(i + 1)[0] for i in range(ncomps)])
        self.hs_model = hs_model(comps, self.eos)
        self.disp_kernel_scale = 1.3862 # Fitted parameter from Sauer and Gross (2016)
        self.computed_weights = {} # lazy evaluation in get_weights(T)

        self.__dispersion_active__ = True
        self.__association_active__ = (self.eos.get_n_assoc_sites() > 0)
        self.__multipole_active__ = True
        self.__chain_active__ = any(self.ms != 1)
        self.__contributions__ = {'dispersion' : self.__dispersion_active__, 'association' : self.__association_active__,
                                  'multipole' : self.__multipole_active__, 'chain' : self.__chain_active__}

    def __repr__(self):
        """Internal
        Called from inheriting classes.

        Returns:
            str : id. string with parameters, active contributions and hard sphere model identifier.
        """
        ostr = f'using parameters : {[self.eos.get_pure_fluid_param(i + 1) for i in range(self.ncomps)]}\n' \
               f'with active contributions : {[k + " : " + str(self.__contributions__[k]) for k in sorted(self.__contributions__.keys())]}\n' \
               f'Hard sphere model : {repr(self.hs_model)}'
        return ostr

    def get_caching_id(self):
        """Utility
        See Functional for docs.
        """
        ostr = f'SAFT : {self._comps}, \n' \
               f'params : {[self.eos.get_pure_fluid_param(i + 1) for i in range(self.ncomps)]}\n' \
               f'kij : {[[self.eos.get_kij(i + 1, j + 1) for i in range(self.ncomps)] for j in range(self.ncomps)]}\n' \
               f'contribs : {[k + " : " + str(self.__contributions__[k]) for k in sorted(self.__contributions__.keys())]}\n' \
               f'HS model : \n{self.hs_model.get_caching_id()}\n'
        return ostr

    def refresh_hs_model(self): # Call this when parameters have been changed by set-methods
        """Internal
        Update hard-sphere model such that parameters are in sync.
        """
        self.ms = np.array([self.eos.get_pure_fluid_param(i + 1)[0] for i in range(self.ncomps)])
        self.hs_model = self.__HS_MODEL__(self._comps, self.eos)

    def get_sigma(self, ic):
        """Utility
        Get the sigma parameter

        Args:
            ic (int) : Component index (zero-indexed)

        Returns:
            float : Model sigma-parameter [m]
        """
        return self.eos.get_pure_fluid_param(ic + 1)[1]

    def set_sigma(self, ic, sigma):
        """Utility
        Set the model sigma-parameter

        Args:
            ic (int) : Component index (zero-indexed)
            sigma (float) : sigma-parameter [m]
        """
        pure_params = list(self.eos.get_pure_fluid_param(ic + 1))
        pure_params[1] = sigma
        self.eos.set_pure_fluid_param(ic + 1, *pure_params)
        self.refresh_hs_model()

    def get_eps_div_k(self, ic):
        """Utility
        Get the epsilon parameter

        Args:
            ic (int) : Component index (zero-indexed)

        Returns:
            float : Model epsilon-parameter, divided by Boltzmanns constant [K]
        """
        return self.eos.get_pure_fluid_param(ic + 1)[2]

    def set_eps_div_k(self, ic, eps_div_k):
        """Utility
        Set the model epsilon-parameter

        Args:
            ic (int) : Component index (zero-indexed)
            eps_div_k (float) : epsilon-parameter divided by Boltzmanns constant [K]
        """
        pure_params = list(self.eos.get_pure_fluid_param(ic + 1))
        pure_params[2] = eps_div_k
        self.eos.set_pure_fluid_param(ic + 1, *pure_params)
        self.refresh_hs_model()

    def set_pure_assoc_param(self, ic, eps, beta):
        """Utility
        Set pure-conponent association parameters

        Args:
            ic (int) : Component index (zero indexed)
            eps (float) : Association energy [J / mol]
            beta (float) : Associaiton volume [-]
        """
        self.eos.set_pure_assoc_param(ic + 1, eps, beta)

    def set_pure_fluid_param(self, ic, m, sigma, eps_div_k, *assoc_param):
        """Utility
        Set all pure component parameters

        Args:
            ic (int) : Component index (zero indexed)
            m (float) : Segment number [-]
            sigma (float) : sigma-parameter [m]
            eps_div_k (float) : epsilon-parameter [K]
        """
        self.set_sigma(ic, sigma)
        self.set_segment_number(ic, m)
        self.set_eps_div_k(ic, eps_div_k)
        if len(assoc_param) > 0:
            self.set_pure_assoc_param(ic, *assoc_param)

    def set_segment_number(self, ic, m):
        """Utility
        Set the segment number

            ic (int) : Component index (zero indexed)
            m (float) : Segment number
        :return:
        """
        pure_params = list(self.eos.get_pure_fluid_param(ic + 1))
        pure_params[0] = m
        self.ms[ic] = m
        self.eos.set_pure_fluid_param(ic + 1, *pure_params)
        self.refresh_hs_model()

    def pair_potential(self, i, j, r):
        """Utility
        Evaluate the pair potential between component `i` and `j` at distance `r`

            i (int) : Component index
            j (int) : Component index
            r (float) : Distance [m]

        Returns:
            float : Interaction potential energy [J]
        """
        # Must be overridden in saft-vrq-mie class to also supply temperature
        return Boltzmann * self.eos.potential(i + 1, j + 1, r * 1e-10, 300)

    def set_dispersion_active(self, active):
        """Utility
        Toggle dispersion contribution on/off

        Args:
            active (bool) : Whether dispersion is active
        """
        self.__dispersion_active__ = active
        self.__contributions__['dispersion'] = active

    def set_association_active(self, active):
        """Utility
        Toggle association contribution on/off

        Args:
            active (bool) : Whether association is active
        """
        self.__association_active__ = active
        self.__contributions__['association'] = active

    def set_chain_active(self, active):
        """Utility
        Toggle chain contribution on/off

        Args:
            active (bool) : Whether chain is active
        """
        self.__chain_active__ = active
        self.__contributions__['chain'] = active

    def set_multipole_active(self, active):
        """Utility
        Toggle multipole contribution on/off

        Args:
            active (bool) : Whether multipole is active
        """
        self.__multipole_active__ = active
        self.__contributions__['multipole'] = active

    def get_weights(self, T, dwdT=False):
        """Weights
        Get all the weights used for weighted densities in a 2D array, indexed as weight[<wt idx>][<comp idx>],
        where weight[:6] are the FMT weights, (w0, w1, w2, w3, wv1, wv2), and weight[6] is the list of
        dispersion weights (one for each component).

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
        dd_dT *= 1e10

        if dwdT is False:
            w_disp = self.get_dispersion_weights(T, d=d, dd_dT=dd_dT)
            weights = self.hs_model.get_weights(T, R=(d / 2))
            weights.extend(w_disp)
            self.computed_weights[T] = copy.deepcopy(weights)
            return weights


        dw_disp_dT = self.get_dispersion_weights(T, dwdT=True, d=d, dd_dT=dd_dT)
        dwdT = self.hs_model.get_weights(T, dwdT=True, R=(d / 2), dRdT=(dd_dT / 2))
        dwdT.extend(dw_disp_dT)
        return dwdT

    def get_dispersion_weights(self, T, dwdT=False, d=None, dd_dT=None):
        """Weights
        Get the weights for the dispersion term

        Args:
            T (float) : Temperature [K]
            dwdT (bool, optimal) : Compute derivative wrt. T? Defaults to False.
            d (1d array) : Pre-computed Barker-Henderson diameters
            dd_dT (1d array) : Pre-computed temperature derivatives of Barker-Henderson diameters.

        Returns:
            2d array of WeightFunction : The weights for the dispersion weighted densities, indexed as wts[<wt_idx>][<comp_idx>]
        """
        if (d is None) or (dd_dT is None):
            d, dd_dT = self.eos.hard_sphere_diameters(T)
            d *= 1e10
            dd_dT *= 1e10

        if dwdT is False:
            w_disp = [[0 for _ in range(self.ncomps)] for _ in range(self.ncomps)]
            for ci in range(self.ncomps):
                w_disp[ci][ci] = NormTheta(d[ci] * self.disp_kernel_scale)
            return w_disp

        dwdT = [[0 for _ in range(self.ncomps)] for _ in range(self.ncomps)]
        for ci in range(self.ncomps):
            dwdT[ci][ci] = NormTheta_diff(d[ci] * self.disp_kernel_scale) * dd_dT[ci]
        return dwdT

    def get_fmt_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False):
        """Weighted density
        Compute weighted densities from FMT model.

        Args:
            rho (list[Profile] or list[float]) : Particle densities of each species
            T (float) : Temperature [K]
            bulk (bool) : Default False, set to True if `rho` is `list[float]`
            dndrho (bool) : Also compute derivatives (only for bulk)
            dndT (bool) : Also compute derivatives

        Returns:
            list[Profile] or list[float] : The weighted densities and optionally *one* differential.
        """
        return self.hs_model.get_weighted_densities(rho, T, bulk=bulk, dndrho=dndrho, dndT=dndT)

    def get_fmt_weights(self, T, dwdT=False):
        """Weights
        Get the FMT weight functions

        Args:
            T (float) : Temperature [K]
            dwdT (bool) : Compute derivative instead
        Returns:
            list[list[WeightFunction]] : FMT weight functions, indexed as wts[<wt_idx>][<comp_idx>]
        """
        return self.hs_model.get_weights(T, dwdT=dwdT)

    def get_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False):
        """Weighted density
        Compute the weighted densities.

        Remember: dphidn is a list[Profile], indexed as dphidn[<weight idx>][<grid idx>]
        Where <weight idx> gives the weight that was used to compute the weighted density.
        Remember also : Each weight gives rise to ONE weighted density, becuase you sum over the components.
                          The exception is the dispersion weight, which gives rise to ncomps weighted densities,
                          one for each component.
        """
        if dndrho is True: # Only if bulk is True
            n_hs, dndrho_hs = self.hs_model.get_weighted_densities(rho, T, bulk=bulk, dndrho=dndrho)
            n_disp, dndrho_disp = self.get_dispersion_weighted_density(rho, T, bulk=bulk, dndrho=dndrho)
            n = [*n_hs, *n_disp]
            dndrho = [*dndrho_hs, *dndrho_disp]
            return n, dndrho
        elif dndT is True:
            n_hs, dndT_hs = self.hs_model.get_weighted_densities(rho, T, bulk=bulk, dndT=dndT)
            n_disp, dndT_disp = self.get_dispersion_weighted_density(rho, T, bulk=bulk, dndT=dndT)
            n = [*n_hs, *n_disp]
            dndT = dndT_hs + dndT_disp
            return n, dndT
        else:
            n_hs = self.hs_model.get_weighted_densities(rho, T, bulk=bulk, dndrho=dndrho)
            n_disp = self.get_dispersion_weighted_density(rho, T, bulk=bulk, dndrho=dndrho)
            n = [*n_hs, *n_disp]
            return n

    def get_dispersion_weighted_density(self, rho_arr, T, bulk=False, dndrho=False, dndT=False):
        """Weighted density
        Get the weighted density for the dispersion term

        Args:
            rho_arr (list[Profile]) : Density profiles [particles / Å^3], indexed as rho_arr[<comp idx>][<grid idx>]
            T (float) : Temperature [K]
            bulk (bool) : For bulk, no convolution is required (this weighted density equals the bulk density)
            dndrho (bool) : Whether to compute the derivatives (all are equal to unity)
            dndT (bool) : Compute the derivatives

        Returns:
            list[Profile] : Weighted densities, indexed as n[<comp idx>][<grid idx>]
            optionally 2d array of ones (dndrho)
        """
        if bulk is True:
            n_disp = rho_arr

        else:
            w_disp = self.get_dispersion_weights(T)
            n_disp = Profile.zeros_like(rho_arr)
            for i, rho_i in enumerate(rho_arr):
                n_disp[i] = Profile(convolve_ad(w_disp[i][i], rho_i), rho_i.grid)

        if dndrho is True:
            dndrho = np.identity(self.ncomps)
            return n_disp, dndrho

        elif dndT is True:
            dwdT = self.get_dispersion_weights(T, dwdT=True)
            dndT = Profile.zeros_like(rho_arr)
            for i, (dwi_dT, rho_i) in enumerate(zip(dwdT, rho_arr)):
                dndT[i] = Profile(convolve_ad(dwi_dT[i][i], rho_i), rho_i.grid)
            return n_disp, dndT

        return n_disp

    def reduce_temperature(self, T, c=0):
        """Utility
        Compute the reduced temperature (LJ units)

        Args:
            T (float) : Temperature [K]
            c (int) : Component idx to use
        Return:
            float : Temperature in LJ units
        """
        _, _, eps_div_k, _, _ = self.eos.get_pure_fluid_param(c + 1)
        z = np.zeros(self.ncomps) + 1e-3
        z[c] = 1 - (self.ncomps - 1) * 1e-3
        Tc = self.eos.critical(z)[0]
        return T / Tc

    def dispersion_helmholtz_energy_density(self, rho, T, bulk=False, dphidn=False, dphidT=False, dphidrho=False, n_disp=None):
        """Helmholtz contribution

        Args:
            rho (list[Profile] or list[float]) : Particle density for each species
            T (float) : Temperature [K]
            bulk (bool) : Default False. Set to True if `rho` is `list[float]`
            dphidn (bool) : Compute derivative
            dphidT (bool) : Compute derivative
            dphidrho (bool) : Compute derivative
            n_disp (list[Profile], optional) : Pre-computed weighted densities.

        Returns:
            Profile or float : The (reduced) dispersion helmholtz energy density [-]
        """
        if dphidT is True:
            n_disp, dndT_disp = self.get_dispersion_weighted_density(rho, T, bulk=bulk, dndT=True)
        elif n_disp is None:
            n_disp = self.get_dispersion_weighted_density(rho, T, bulk=bulk)

        if dphidrho is True:
            phi, dphidn = self.dispersion_helmholtz_energy_density(rho, T, dphidn=True, bulk=bulk)
            w = self.get_dispersion_weights(T)
            if bulk is True:
                dphidrho = np.zeros(self.ncomps)
                for i in range(self.ncomps):
                    dphidrho[i] += dphidn[i] * w[i][i].real_integral()
            else:
                dphidrho = Profile.zeros_like(rho)
                for i in range(self.ncomps):
                    dphidrho[i] += convolve_ad(w[i][i], dphidn[i])

            return phi, dphidrho


        V = 1.0  # So that the densities can be treated as mole numbers when thermopack is called
        if bulk is True:  # When computing chemical potential bulk is True.
            # See below for explanation regarding unit conversions.
            n_disp = [n / (Avogadro * 1e-30) for n in n_disp]
            if dphidn is True:
                a, a_n = self.eos.a_dispersion(T, V, n_disp, a_n=True)
            elif dphidT is True:
                a, a_T, a_n = self.eos.a_dispersion(T, V, n_disp, a_t=True, a_n=True)
            else:
                a, = self.eos.a_dispersion(T, V, n_disp)

            n_tot = sum(n_disp) * Avogadro * 1e-30
            phi_disp = n_tot * a

            if dphidn is True:
                dphidn_disp = np.zeros(self.ncomps)
                for ci in range(self.ncomps):
                    dphidn_disp[ci] = a + n_tot * a_n[ci] / (Avogadro * 1e-30)

                return phi_disp, dphidn_disp

            if dphidT is True:
                dn_tot_dT = sum(dndT_disp)
                dphidT_disp = a * dn_tot_dT
                for ci in range(self.ncomps):
                    dphidT_disp += (a_n[ci] / (Avogadro * 1e-30)) * dndT_disp[ci]

                return phi_disp, dphidT_disp


            return phi_disp

        phi_disp = Profile.zeros(n_disp[0].grid)  # Returns Profile, with grid taken from n_disp
        if dphidn is True:
            dphidn_disp = Profile.zeros_like(n_disp)  # Returns list[Profile] with same shape as n_disp, and grid taken from n_disp
        elif dphidT is True:
            dphidT_disp = Profile.zeros(n_disp[0].grid)

        for ri in range(len(phi_disp)):  # Iterate over the grid positions
            n = [n_disp[ci][ri] / (Avogadro * 1e-30) for ci in range(self.ncomps)]  # Convert from [particles / Å^3] to [mol / m^3]
            n_tot = sum(n) * Avogadro * 1e-30  # Total density in [particles / Å^3]

            if abs(n_tot) < 1e-12:  # Ensure that thermopack does not return NaN
                a, a_n = 0, np.zeros(self.ncomps)
            else:
                if dphidn is True:
                    a, a_n = self.eos.a_dispersion(T, V, n, a_n=True)  # Gives nan if sum(n) == 0
                elif dphidT is True:
                    a, a_T, a_n = self.eos.a_dispersion(T, V, n, a_t=True, a_n=True)
                else:
                    a, = self.eos.a_dispersion(T, V, n)

            """
            Thermopack returns the reduced *Molar* Helmholtz energy, i.e. (A / n R T), so by multiplying with the
            particle density we get the reduced Helmholtz energy density in [1 / Å^3]. In the following, note that

            a *= R * T # Multiplying by [(J / (mol K)) * K) to get molar Helmholtz energy [J / mol]
            a /= Avogdro # Converting to [J / particle]
            a *= n_tot # Converting to [J / Å^3] (n_tot has unit [particles / Å^3], see above
            a /= (Boltzmann * T) # Converting to [1 / Å^3]

            is equivalent to just doing

            a *= n_tot # Converting to [1 / Å^3]

            Furthermore, a_n has unit [m^3 / mol] from thermopack (because we compute for V = 1.0 ??),
            but should be dimensionless here
            (phi has unit [1 / Å^3] and the weighted densities (n) have unit [1 / Å^3])

            So we do:

            a_n *= R * T # Converting to [J m^3 / mol^2]
            a /= Avogadro # Converting to [J m^3 / (particle * mol)]
            a *= n_tot # Converting to [J m^3 / (Å^3 mol)]
            a /= Boltzmann * T # Converting to [m^3 / (Å^3 mol)]
            a /= Avogadro # Converting to [m^3 / Å^3] = [ 1e-30 ]
            a /= 1e-30 # Converting to [–]

            which is equivalent to

            a_n *= n_tot / (Avogadro * 1e-30)
            """
            phi_disp[ri] = n_tot * a
            if dphidn is True:
                for ci in range(self.ncomps):
                    dphidn_disp[ci][ri] = a + n_tot * a_n[ci] / (Avogadro * 1e-30)
            elif dphidT is True:
                dn_tot_dT = 0
                for ci in range(self.ncomps):
                    dn_tot_dT += dndT_disp[ci][ri]

                dphidT_disp[ri] = dn_tot_dT * a + n_tot * a_T
                for ci in range(self.ncomps):
                    dphidT_disp[ri] += n_tot * (a_n[ci] / (Avogadro * 1e-30)) * dndT_disp[ci][ri]

        if dphidn is True:
            return phi_disp, dphidn_disp
        elif dphidT is True:
            return phi_disp, dphidT_disp
        return phi_disp

    def reduced_helmholtz_energy_density(self, rho, T, dphidn=False, bulk=False, asarray=False, dphidT=False):
        """Profile Property
        Compute the reduced helmholtz energy density [1 / Å^3] (see the Functional class for explanation, this is simply
        an overlaod that puts toghether all the contributions.)

        Args:
            rho (list[Profile] or list[float]) : Density profiles [particles / Å^3], indexed as rho[<comp idx>][<grid idx>].
                                                    Takes list[Profile] if `bulk is False`, and list[float] if `bulk is True`.
            T (float) : Temperature [K]
            dphidn (bool) : Whether to compute differentials wrt. weighted densities
            bulk (bool) : passed on to get_weighted_density, because convolutions are not neccesary for bulk.
                            If True, rho should be list[float].
            asarray (bool) : Do not set to True (I dont think that works yet)
                            Intent: return helmholtz density as a 1d numpy array rather than a Profile
            dphidT (bool) : Compute temperature derivative

        Returns:
            Profile : Reduced Helmholtz energy density [1 / Å^3]
            Optionally list[Profile] : derivatives wrt. weighted densities.
        """
        # First, compute hard sphere contribution by passing the call to the hs_model
        if dphidn is True:
            phi_hs, dphidn_hs = self.hs_model.reduced_helmholtz_energy_density(rho, T, dphidn=dphidn, bulk=bulk, asarray=False)
            phi_disp, dphidn_disp = self.dispersion_helmholtz_energy_density(rho, T, bulk=bulk, dphidn=dphidn)
            # Remember: dphidn is a list[Profile], indexed as dphidn[<weight idx>][<grid idx>]
            # Where <weight idx> gives the weight that was used to compute the weighted density.
            # Remember also : Each weight gives rise to ONE weighted density, becuase you sum over the components.
            #                   The exception is the dispersion weight, which gives rise to ncomps weighted densities,
            #                   one for each component.
            dphidn_hs.extend(dphidn_disp)
            phi = phi_hs + phi_disp
            return phi, dphidn_hs
        elif dphidT is True:
            phi_hs, dphidT_hs = self.hs_model.reduced_helmholtz_energy_density(rho, T, dphidT=dphidT, bulk=bulk, asarray=False)
            phi_disp, dphidT_disp = self.dispersion_helmholtz_energy_density(rho, T, bulk=bulk, dphidT=dphidT)
            phi = phi_hs + phi_disp
            dphidT = dphidT_hs + dphidT_disp
            return phi, dphidT
        else:
            phi_hs = self.hs_model.reduced_helmholtz_energy_density(rho, T, bulk=bulk, asarray=False)
            phi_disp = self.dispersion_helmholtz_energy_density(rho, T, bulk=bulk)
            phi = phi_hs + phi_disp
            return phi

    def pressure_tv(self, rho, T):
        """Deprecated
        """
        pass