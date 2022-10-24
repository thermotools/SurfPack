#!/usr/bin/env python3
from collections import OrderedDict
from enum import Enum

# Debug flag
DEBUG = False

# Avogadro's number
NA = 6.02214076e23
# Boltzmann number
KB = 1.380650524e-23
# Gas constant
RGAS = NA*KB
# Planck’s constant
h = 6.626069311e-34

class DftEnum(Enum):
    def __eq__(self, other):
        #print("isinstance",other.__class__, self.__class__, isinstance(other, self.__class__))
        #if isinstance(other, self.__class__):
        try:
            return self.value == other.value
        except AttributeError:
            return False

class Geometry(DftEnum):
    PLANAR = 1
    POLAR = 2
    SPHERICAL = 3

class Specification(DftEnum):
    NUMBER_OF_MOLES = 10
    CHEMICHAL_POTENTIAL = 20

class LenghtUnit(DftEnum):
    ANGSTROM = 100
    REDUCED = 200

class ProfileInit(DftEnum):
    TANH = 1000
    PREVIOUS_SCALED = 2000

class Properties(DftEnum):
    RHO = 11
    FREE_ENERGY = 12
    ENERGY = 13
    ENTROPY = 14
    ENTHALPY = 15
    CHEMPOT_SUM = 16
    PARALLEL_PRESSURE = 17
    CHEMPOT = 18
    CHEMPOT_ID = 19
    CHEMPOT_EX = 21

# Plot utility
LINE_STYLES = OrderedDict(
    [('0',               (0, ())),
     ('1',              (0, (5, 5))),
     ('2',              (0, (1, 5))),
     ('3',          (0, (3, 5, 1, 5))),
     ('4',         (0, (3, 5, 1, 5, 1, 5))),
     ('5',      (0, (1, 10))),
     ('6',      (0, (5, 10))),
     ('7',  (0, (3, 10, 1, 10))),
     ('8', (0, (3, 10, 1, 10, 1, 10))),
     ('9',      (0, (1, 1))),
     ('10',      (0, (5, 1))),
     ('11',  (0, (3, 1, 1, 1))),
     ('12', (0, (3, 1, 1, 1, 1, 1)))])

LCOLORS = ["k", "g", "r", "b", "grey", "m", "c", "y",
           "tomato", "peru", "olive", "royalblue", "darkviolet"]
LMARKERS = ["*", "d", "o", "s", "+", "x", "1", "h", "2", "v", "p", "^", "X"]

def get_property_label(prop, reduced, ic=0):
    if prop == Properties.RHO:
        label = r"$\rho^*$" if reduced else r"$\rho$ (mol/m$^3$)"
    elif prop == Properties.FREE_ENERGY:
        label = r"$a_{\rm{E}}^*$" if reduced else r"$a_{\rm{E}}$ (J/m$^3$)"
    elif prop == Properties.ENERGY:
        label = r"$e_{\rm{E}}^*$" if reduced else r"$e_{\rm{E}}$ (J/m$^3$)"
    elif prop == Properties.ENTROPY:
        label = r"$s_{\rm{E}}^*$" if reduced else r"$s_{\rm{E}}$ (J/m$^3$)"
    elif prop == Properties.ENTHALPY:
        label = r"$h_{\rm{E}}^*$" if reduced else r"$h_{\rm{E}}$ (J/m$^3$)"
    elif prop == Properties.CHEMPOT_SUM:
        label = r"CHEMPOT_SUM"
    elif prop == Properties.PARALLEL_PRESSURE:
        label = r"$p_{\parallel}^*$" if reduced else r"$p_{\parallel}$ (MPa)"
    elif prop == Properties.CHEMPOT:
        label = "$\\mu^*_{\\rm{"+str(ic+1)+"}}$" if reduced else "$\mu_{\\rm{"+str(ic+1)+"}}$ (J/mol)"
    elif prop == Properties.CHEMPOT_ID:
        label = "$\mu_{\rm{I}}^*_{\\rm{"+str(ic+1)+"}}$" if reduced else "$\mu_{\rm{I}}_{\\rm{"+str(ic+1)+"}}$ (J/mol)"
    elif prop == Properties.CHEMPOT_EX:
        label = "$\mu_{\rm{E}}^*_{\\rm{"+str(ic+1)+"}}$" if reduced else "$\mu_{\rm{E}}_{\\rm{"+str(ic+1)+"}}$ (J/mol)"
    else:
        raise ValueError("Wrong property in get_property_label")
    return label

if __name__ == "__main__":

    pass
