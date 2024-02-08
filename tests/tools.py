FLTEPS = 1e-10
def is_equal(a, b):
    return abs(a - b) < FLTEPS

def is_equal_arr(a, b):
    return all(abs(a - b) < FLTEPS)

singlecomps = ['KR', 'NC6', 'NH3']
singlecomps2 = ['AR', 'C2', 'H2O']
binaries = ['KR,AR', 'KR,C2', 'C3,NC6', 'NC6,NH3', 'NH3,H2O']