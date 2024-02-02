import numpy as np
from scipy.constants import Boltzmann
import abc

"""
The computations done elsewhere often use an external potential, which is a callable.

The classes here are intended to be used as templates to generate those callables.

Per now, the code assumes that the callables return a single value, but should probably be updated to 
return one value for each component (or be agnostic, such that e.g. HardWall can return a single number, while
LennardJones93 can return one value per compnent).
"""
class ExtPotential(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, *args, **kwargs): pass

    @abc.abstractmethod
    def __hash__(self): pass

class HardWall(ExtPotential):

    def __init__(self, position, is_pore=True):
        """
        is_pore determines whether the "wall" is at
            True => (position, inf) or
            False => (-inf, position)
        """
        self.is_pore = is_pore
        self.position = position

    def __call__(self, z):
        V = np.zeros_like(z)
        if self.is_pore is True:
            V[z > self.position] += np.inf
        else:
            V[z < self.position] += np.inf

        return V

    def __hash__(self):
        return hash(('HardWall', self.position, self.is_pore))

class SlitPore(ExtPotential):

    def __init__(self, width, potential):
        self.Vext = potential
        self.width = width

    def __call__(self, z):
        return self.Vext(z) + self.Vext(self.width - z)

    def __hash__(self):
        return hash(('SoftSlitPore', self.width, hash(self.Vext)))

class LennardJones93(ExtPotential):

    def __init__(self, sigma, epsilon):
        """
        This was modified just as I got stuff working, but the commented out solution is probably better,
        i.e. computing sigma and epsilon from mixing rules, such that the potential can be assymetric (see top)
        """
        self.sigma = sigma # 0.5 * (sigma + func.eos.sigma * 1e10)
        self.epsilon = epsilon # np.sqrt(epsilon * func.eos.eps_div_kb) * Boltzmann

    def __call__(self, z):
        return self.epsilon * ((self.sigma / z) ** 9 - (self.sigma / z) ** 3)

    def __hash__(self):
        return hash(('LennardJones93', self.sigma, self.epsilon))

class Steele(ExtPotential):

    def __init__(self, sigma, epsilon, delta, rho_s):
        self.sigma, self.epsilon, self.delta, self.rho_s = sigma, epsilon, delta, rho_s

    def __call__(self, z):
        return 2 * np.pi * self.rho_s * self.epsilon * self.sigma**2 * self.delta \
               * ((2 / 5) * (self.sigma / z)**10 - (self.sigma / z)**4 - self.sigma**4 / (3 * self.delta * (z + 0.61 * self.delta)**3))

    def __hash__(self):
        return hash(('Steele', self.sigma, self.epsilon, self.delta, self.rho_s))

class Mie(ExtPotential):

    def __init__(self, sigma, eps_div_k, lr, la):
        self.sigma = sigma
        self.epsilon = eps_div_k * Boltzmann
        self.lr = lr
        self.la = la
        self.C = (lr / (lr - la)) * (lr / la) ** (la / (lr - la))

    def __call__(self, z):
        return self.C * self.epsilon * ((self.sigma / z)**self.lr - (self.sigma / z)**self.la)

    def __hash__(self):
        return hash(('Mie', self.sigma, self.epsilon, self.la, self.lr))

class ModifiedSquareWell(ExtPotential):

    def __init__(self, sigma, eps_div_k):
        self.sigma = sigma
        self.s = 0.12 * sigma
        self.epsilon = eps_div_k * Boltzmann

    def __call__(self, z):
        V = np.zeros_like(z)
        V[z < self.sigma - self.s] = np.inf
        V[(self.sigma - self.s < z) & (z < self.sigma)] = 3 * self.epsilon
        V[(self.sigma < z) & (z < self.sigma + self.s)] = - self.epsilon
        return V

    def __hash__(self):
        return hash(('ModifiedSquareWell', self.sigma, self.epsilon, self.s))

class ExpandedSoft(ExtPotential):

    def __init__(self, R, soft_potential):
        self.R = R
        self.outer = soft_potential

    def __call__(self, z):
        V = np.zeros_like(z)
        V[z <= self.R] = np.inf
        V[z > self.R] = self.outer(z - self.R)[z > self.R]
        return V

    def __hash__(self):
        return hash(('ExpandedSoft', self.R, hash(self.outer)))

