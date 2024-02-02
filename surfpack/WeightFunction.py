"""
The weight functions are implemented as callable classes, with an __mul__ implemented such that for example

2 * Heaviside(R) # Returns a new callable

See the Analytical class for more info.
"""

import numpy as np
from scipy.special import spherical_jn

class Analytical:
    """
    Parent class for analytical functions
    Intended to be used for the fourier transformed weight functions
        The (fourier transform) of the weight function is implemented in the self.lamb attribute, and called from the __call__ method.

    Implements __mul__, such that multiplying an `Analytical` object with a float, returns a new `Analytical` object
    This makes generating weight functions very clean and easy.

    Also implements the `is_even()` and `is_odd()` methods, which return whether or not the function is even or odd,
    for heaviside and delta functions, this is as simple as checking whether the function is_vector_valued, in which case
    it is odd. These checks are used when computing convolutions to select the appropriate sin/cos transforms.

    Example:
        f = Analytical(lambda x : x**2 - x)
        g = 2 * f
        f(3) # Returns 6 (= 3**2 - 3)
        g(3) # Returns 12 (= 2 * (3**2 - 3))
    """
    def __init__(self, lamb, integral, is_vector_valued=False):
        self.lamb = lamb
        self.integral = integral
        self.is_vector_valued = is_vector_valued

    def __call__(self, k):
        return self.lamb(k)

    def __mul__(self, prefactor): # Used to dynamically generate weight functions - See comment at top of class
        return Analytical(lambda k : prefactor * self(k), prefactor * self.real_integral(), self.is_vector_valued)

    def __rmul__(self, prefactor):
        return self.__mul__(prefactor)

    def __truediv__(self, other):
        return self * (1 / other)

    def __sub__(self, other):
        return Analytical(lambda k: self(k) - other(k), self.real_integral() - other.real_integral(), self.is_vector_valued)

    def __rsub__(self, other):
        return Analytical(lambda k: other(k) - self(k), other.real_integral - self.real_integral(), self.is_vector_valued)

    def __add__(self, other):
        return Analytical(lambda k: self(k) + other(k), self.real_integral() + other.real_integral(), self.is_vector_valued)

    def __radd__(self, other):
        return self.__add__(other)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False

        raise TypeError("Equality between analytical functions is not implemented.")

    def is_odd(self):
        return self.is_vector_valued

    def is_even(self):
        return not self.is_odd()

    def real_integral(self):
        return self.integral

class LocalDensity(Analytical):
    """
    Not an actual weight, but a placeholder to indicate we are using the local density. Treated specially in Convolver.py
    """
    def __init__(self, mult_factor=1):
        self.mult_factor = mult_factor
        super().__init__(lambda k: None, self.mult_factor)

    def __mul__(self, other):
        return LocalDensity(self.mult_factor * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return LocalDensity(self.mult_factor / other)

    def __call__(self, k):
        raise AttributeError('The local density does not have a weight function! This is a placeholder!')

class Heaviside(Analytical):
    """
    3D Fourier transform of $\theta(r - R)$
    """
    def __init__(self, kernel):
        self.R = kernel
        super().__init__(lambda k: (4/3) * np.pi * self.R**3 * (spherical_jn(0, 2 * np.pi * k * self.R) + spherical_jn(2, 2 * np.pi * k * self.R)), 4 * np.pi * self.R**3 / 3)

class Heaviside_diff(Analytical):
    """
    Derivative of Heaviside wrt. R
    """

    def __init__(self, kernel):
        self.R = kernel
        super().__init__(lambda k: 4 * np.pi * self.R ** 2 * spherical_jn(0, 2 * np.pi * k * self.R), 4 * np.pi * self.R**2)

class NormTheta(Analytical):
    """
    3D Fourier transform of $((4/3) * np.pi * R^3)^{-1} \theta(r - R)$
    """
    def __init__(self, kernel):
        self.R = kernel
        super().__init__(lambda k: spherical_jn(0, 2 * np.pi * k * self.R) + spherical_jn(2, 2 * np.pi * k * self.R), 1)

class NormTheta_diff(Analytical):

    def __init__(self, kernel):
        self.R = kernel
        super().__init__(lambda k: - (3 / self.R) * spherical_jn(2, 4 * np.pi * self.R * k), 3 / self.R)

class Delta(Analytical):
    r"""
    3D Fourier transform of $\delta(r - R)$
    """
    def __init__(self, kernel):
        self.R = kernel
        super().__init__(lambda k: 4 * np.pi * self.R**2 * spherical_jn(0, 2 * np.pi * k * self.R), 4 * np.pi * self.R**2)

class Delta_diff(Analytical):
    """
    Derivative of fourier transform of delta function wrt. R.
    """
    def __init__(self, kernel):
        self.R = kernel
        super().__init__(lambda k: 8 * np.pi * self.R * (spherical_jn(0, 2 * np.pi * self.R * k)
                                                         - np.pi * self.R * k * spherical_jn(1, 2 * np.pi * self.R * k)), 0)

class DeltaVec(Analytical):
    r"""
    3D Fourier transform of $\hat{\vec{r}}\delta(r - R)$, where $\hat{\vec{r}} = \vec{r} / |\vec{r}|$ is the unit vector
    pointing away from the origin.
    """
    def __init__(self, kernel):
        self.R = kernel
        super().__init__(lambda k: - 2 * np.pi * k * 4.0 / 3.0 * np.pi * self.R ** 3 \
                        * (spherical_jn(0, 2 * np.pi * k * self.R) + spherical_jn(2, 2 * np.pi * k * self.R)),
                         0., is_vector_valued=True)

class DeltaVec_diff(Analytical):
    """
    Derivative of DeltaVec wrt. R
    """
    def __init__(self, kernel):
        self.R = kernel
        super().__init__(lambda k: - 8 * np.pi**2 * k * self.R**2 * spherical_jn(0, 2 * np.pi * self.R * k), 0)


def get_FMT_weights(R, ms=None, as_dict=False):
    """
    Return an array of `Analytical` weight functions, organised as

    w[<weight index>][<component index>], where
    w[0:4] are the scalar weight functions, and
    w[4:6] are the vector weight functions $\vec{w}_1$ and $\vec{w}_2$
    """
    w = [[Analytical(0, 0) for _ in range(len(R))] for _ in range(6)]
    if ms is None:
        ms = np.ones_like(R)
    for i in range(len(R)):
        w[0][i] = ms[i] * (1 / (4 * np.pi * R[i] ** 2)) * Delta(R[i])
        w[1][i] = ms[i] * (1 / (4 * np.pi * R[i])) * Delta(R[i])
        w[2][i] = ms[i] * Delta(R[i])
        w[3][i] = ms[i] * Heaviside(R[i])
        w[4][i] = ms[i] * (1 / (4 * np.pi * R[i])) * DeltaVec(R[i])
        w[5][i] = ms[i] * DeltaVec(R[i])

    if as_dict is False:
        return w
    return {'w0' : w[0], 'w1' : w[1], 'w2' : w[2], 'w3' : w[3], 'wv1' : w[4], 'wv2' : w[5]}


def get_FMT_weight_derivatives(R, as_dict=False):
    """
    See: get_FMT_weights
    This function returns the derivatives wrt. R.
    """
    dwdR = [[Analytical(0) for _ in range(len(R))] for _ in range(6)]
    for i in range(len(R)):
        dwdR[0][i] = (1 / (4 * np.pi * R[i] ** 2)) * Delta_diff(R[i]) - (1 / (2 * np.pi * R[i]**3)) * Delta(R[i])
        dwdR[1][i] = (1 / (4 * np.pi * R[i])) * Delta_diff(R[i]) - (1 / (4 * np.pi * R[i]**2)) * Delta(R[i])
        dwdR[2][i] = Delta_diff(R[i])
        dwdR[3][i] = Heaviside_diff(R[i])
        dwdR[4][i] = (1 / (4 * np.pi * R[i])) * DeltaVec_diff(R[i]) - (1 / (4 * np.pi * R[i]**2)) * Delta(R[i])
        dwdR[5][i] = DeltaVec_diff(R[i])

    if as_dict is False:
        return dwdR
    return {'dw0dR' : dwdR[0], 'dw1dR' : dwdR[1], 'dw2dR' : dwdR[2], 'dw3dR' : dwdR[3], 'dwv1dR' : dwdR[4], 'dwv2dR' : dwdR[5]}