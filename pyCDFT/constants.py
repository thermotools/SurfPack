#!/usr/bin/env python3
from collections import OrderedDict

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


if __name__ == "__main__":

    pass
