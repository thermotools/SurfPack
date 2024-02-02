"""
This is the module that handles all the special treatment of functions depending on geometry, symmetry, etc.

The basis is the 'convolve_ad' function, which takes an Analytical object (WeightFunction.py) and a Profile (Profile.py).

By looking at the Profile's 'grid' attribute (specifically the grid.geometry attribute), as well as the is_even() and is_odd()
functions of the Profile and the Analytical, this module determines what series of transforms to use to compute the convolution.

For the math: See the dft.pdf in the docs folder.
"""

from scipy.fft import dst, idst, dct, idct
import numpy as np
from surfpack.grid import Geometry
from surfpack.WeightFunction import LocalDensity

def convolve_ad(analytical, discrete):
    """
    Convolve an analytical function with a discrete function

    Args:
        analytical (Analytical) : Callable returning an analytical, fourier-transformed function. (See: WeightFunction.py for examples)
        discrete (Profile) : The discrete function
    Returns:
        ndarray : A 1d array of the convolved function (in real space)
                    NOTE: Does NOT return a Profile.
    """
    if isinstance(analytical, LocalDensity):
        return discrete * analytical.mult_factor
    elif analytical == 0:
        return discrete * 0
    elif discrete.grid.geometry == Geometry.PLANAR:
        return convolve_ad_planar(analytical, discrete)
    elif discrete.grid.geometry == Geometry.POLAR:
        return convolve_ad_polar(analytical, discrete)
    elif discrete.grid.geometry == Geometry.SPHERICAL:
        return convolve_ad_spherical(analytical, discrete)

def convolve_ad_planar(analytical, discrete):
    """
    Convolutions for planar geometry

    Args:
        analytical (Analytical) : Callable returning an analytical, fourier-transformed function. (See: WeightFunction.py for examples)
        discrete (Profile) : The discrete function
    Returns:
        ndarray : A 1d array of the convolved function (in real space)
                    NOTE: Does NOT return a Profile.
    """

    # Determine the forward transform (ft) and the inverse transform (inv_ft) to be used, based on
    # the even/oddness of the functions.
    # The grid on which to evaluate the analytical function is determined by the forward transform.
    # Whether or not the discrete, transformed function must be "rolled" is determined by whether the forward
    # and inverse transforms are equivalent.
    if discrete.is_even():
        ft = dct

        if analytical.is_even():
            k = discrete.grid.k_cos
            inv_ft = idct
            roll = 0
        else:
            k = discrete.grid.k_sin
            inv_ft = idst
            roll = -1
            remove_idx = -1
    else:
        ft = dst
        if analytical.is_odd():
            k = discrete.grid.k_cos
            inv_ft = lambda x, type=2: - idct(x, type=type)
            roll = +1
            remove_idx = 0
        else:
            k = discrete.grid.k_sin
            inv_ft = idst
            roll = 0

    # Transform (and possibly roll) the discrete function.
    if roll == 0:
        discrete_transformed = ft(discrete, type=2)
    else:
        discrete_transformed = np.roll(ft(discrete, type=2), roll)
        discrete_transformed[remove_idx] = 0

    # Do the convolution
    return inv_ft(discrete_transformed * analytical(k), type=2)

def convolve_ad_polar(analytical, discrete):
    raise NotImplementedError

def convolve_ad_spherical(analytical, discrete):
    """
    Convolutions for spherical geometry

    Args:
        analytical (Analytical) : Callable returning an analytical, fourier-transformed function. (See: WeightFunction.py for examples)
        discrete (Profile) : The discrete function
    Returns:
        ndarray : A 1d array of the convolved function (in real space)
                    NOTE: Does NOT return a Profile.
    """
    r = discrete.grid.z

    # Define shifted profiles, such that discrete_delta[-1] = 0
    discrete_inf = discrete[-1]
    discrete_delta = discrete - discrete_inf

    w_sin = analytical(discrete.grid.k_sin)
    w_cos = analytical(discrete.grid.k_cos)

    k_sin = discrete.grid.k_sin
    k_cos = discrete.grid.k_cos

    # Note : The argument to the convolution is f(r) * r, which is odd if f(r) is even
    if discrete.is_even():
        if analytical.is_even():
            delta_term = (1 / r) * idst(dst(discrete_delta * r) * w_sin)
            inf_term = analytical(0) * discrete_inf
            return delta_term + inf_term
        else:
            odd_term = dst(discrete_delta * r, type=2) * w_sin / k_sin
            even_term = np.roll(dst(discrete_delta * r, type=2) / k_sin, +1) * w_cos * k_cos
            even_term[0] = 0

            delta_term = (1 / (np.pi * r**2)) * idst(odd_term, type=2) / 2 - (1 / r) * idct(even_term, type=2) # Need to divide idst by 2 because of transform prefactor
            return delta_term # Inf term vanishes for odd analytical function (vector weights)

    else:
        if analytical.is_even():
            raise NotImplementedError('Case (a, d) = (even, odd) is not implemented for spherical geometry.')
        else:
            cos_term = np.roll((1 / k_cos) * dct(discrete_delta * r, type=2), -1)
            cos_term[-1] = 0
            sin_term = dst(discrete_delta, type=2) / (np.pi * k_sin**2)

            delta_term = (1 / r) * idst((cos_term - sin_term) * w_sin * k_sin)
            return delta_term




