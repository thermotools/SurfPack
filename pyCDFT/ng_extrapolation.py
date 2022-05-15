#!/usr/bin/env python3
import numpy as np
from pyCDFT.utility import densities


class ng_nc_wrapper():
    """
    Wrap ng_extrapolation for nc components
    """

    def __init__(self, nc, N, n_update,  array_mask):
        """

        Args:
            nc (int): Number of components
            N (int): Size of arrays
            n_update (int): Update only every n_update iteration
            array_mask (np.ndarray of bool): Mask array calculations
        """
        self.nc = nc
        self.array_mask = array_mask
        self.densities = densities(nc, N)
        self.i0 = None
        self.iN = None
        for i in range(N):
            if array_mask[i]:
                self.i0 = i
                break
        for i in range(N):
            if array_mask[N-1-i]:
                self.iN = N-1-i
                break
        self.N_masked = self.iN - self.i0 + 1
        self.fn = np.zeros(self.N_masked*nc)
        self.gn = np.zeros(self.N_masked*nc)
        self.ng = ng_extrapolation(N=self.N_masked*nc, n_update=n_update)

    def push_back(self, fn, gn, iteration):
        """

        Args:
            fn (densities): Variable of iterative method
            gn (densities): gn = A fn
            iteration (int): Current iteration
        """
        for i in range(self.nc):
            self.fn[self.N_masked*i:self.N_masked *
                    (i+1)] = fn[i][self.array_mask]
            self.gn[self.N_masked*i:self.N_masked *
                    (i+1)] = gn[i][self.array_mask]
        self.ng.push_back(self.fn, self.gn, iteration)

    def time_to_update(self, iteration):
        """

        Args:
            iteration (int): Iteration index

        Returns:
            (bool): True if it is time to update
        """
        return self.ng.time_to_update(iteration)

    def extrapolate(self):
        """
        """
        extrapolation = self.ng.extrapolate()
        for i in range(self.nc):
            self.densities[i][self.array_mask] = extrapolation[self.N_masked *
                                                               i:self.N_masked * (i+1)]
        return self.densities


class ng_extrapolation():
    """
    Accelerate Picard solution using extrapolations as described in appendix of Ng (1974):
        Kin‐Chue Ng
        Hypernetted chain solutions for the classical one‐component plasma up to Γ=7000
        The Journal of Chemical Physics
        1974. 61(7):2680-2689
        doi: 10.1063/1.1682399
    """

    def __init__(self, N, n_update, array_mask=None):
        """

        Args:
            N (int): Size of arrays
            n_update (int): Update only every n_update iteration
            array_mask (np.ndarray of bool): Mask array calculations
        """
        self.N = N
        if n_update is not None:
            self.n_update = max(n_update, 3)
        else:
            self.n_update = None
        self.gn = np.zeros(N)
        self.gnm1 = np.zeros(N)
        self.gnm2 = np.zeros(N)
        self.dn = np.zeros(N)
        self.dnm1 = np.zeros(N)
        self.dnm2 = np.zeros(N)
        # Mask calculations
        if array_mask is None:
            self.array_mask = np.full(N, True, dtype=bool)
        else:
            self.array_mask = array_mask

    def push_back(self, fn, gn, iteration):
        """

        Args:
            fn (np.ndarray): Variable of iterative method
            gn (np.ndarray): gn = A fn
            iteration (int): Current iteration
        """
        if self.n_update is None:
            return
        # No need to copy arrays that will not be used:
        if iteration % self.n_update == 0 or iteration % self.n_update > self.n_update - 3:
            self.gnm2[self.array_mask] = self.gnm1[self.array_mask]
            self.gnm1[self.array_mask] = self.gn[self.array_mask]
            self.gn[self.array_mask] = gn[self.array_mask]

            self.dnm2[self.array_mask] = self.dnm1[self.array_mask]
            self.dnm1[self.array_mask] = self.dn[self.array_mask]
            self.dn[self.array_mask] = self.gn[self.array_mask] - \
                fn[self.array_mask]

    def extrapolate(self):
        """
        """
        d01 = self.dn[self.array_mask] - self.dnm1[self.array_mask]
        d02 = self.dn[self.array_mask] - self.dnm2[self.array_mask]
        a = np.zeros((2, 2))
        b = np.zeros(2)
        # Divide by N to avoid too large numbers
        b[0] = np.inner(self.dn[self.array_mask], d01)
        b[1] = np.inner(self.dn[self.array_mask], d02)
        a[0, 0] = np.inner(d01, d01)
        a[0, 1] = np.inner(d01, d02)
        a[1, 0] = a[0, 1]
        a[1, 1] = np.inner(d02, d02)
        scaling = max(np.max(np.abs(b)), 1.0e-3)
        a /= scaling
        b /= scaling
        c = np.linalg.solve(a, b)
        fnp1 = (1 - c[0] - c[1]) * self.gn[self.array_mask] + \
            c[0] * self.gnm1[self.array_mask] + \
            c[1] * self.gnm2[self.array_mask]
        return fnp1

    def time_to_update(self, iteration):
        """

        Args:
            iteration (int): Iteration index

        Returns:
            (bool): True if it is time to update
        """

        if self.n_update is None:
            update = False
        elif iteration < 3:
            update = False
        else:
            update = iteration % self.n_update == 0
        return update


if __name__ == "__main__":
    pass
