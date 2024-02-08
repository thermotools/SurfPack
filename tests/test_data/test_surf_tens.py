from surfpack.pcsaft import PC_SAFT
import pandas as pd
import pytest
from tests.tools import is_equal, is_equal_arr, singlecomps, singlecomps2, binaries

@pytest.mark.parametrize('comp', singlecomps)
def test_singlecomp(comp):
    dft = PC_SAFT(comp)

    gamma, T = dft.surface_tension_singlecomp(n_points=3, t_max=0.85)
    compare_data = pd.read_csv(f'data/surf_tens_{comp}.csv')
    assert is_equal_arr(T, compare_data['T'])
    assert is_equal_arr(gamma, compare_data['gamma'])


def gendata_singlecomp():
    for comp in singlecomps:
        dft = PC_SAFT(comp)

        gamma, T = dft.surface_tension_singlecomp(n_points=3, t_max=0.85)
        pd.DataFrame({'T' : T, 'gamma' : gamma}).to_csv(f'data/surf_tens_{comp}.csv')

@pytest.mark.parametrize('comps', binaries)
@pytest.mark.parametrize('t', [0.7, 0.8, 0.9])
def test_binary(comps, z, t): # Also (implicitly) checks that component order is irrelevant
        comp1, comp2 = comps.split(',')
        compare_data = pd.read_csv(f'data/surf_tens_binary_{comp1}_{comp2}.csv')
        dft = PC_SAFT(','.join((comp1, comp2)))
        Tc = 1e6
        for z in [1e-3, 0.5, 1 - 1e-3]:
            Tc_new = dft.eos.critical([z])[0]
            if Tc_new < Tc:
                Tc = Tc_new
        T = Tc * t
        compare_data = compare_data[compare_data['T'] == T]
        gamma, lve = dft.surface_tension_isotherm(T, n_points=3, calc_lve=True)
        assert is_equal_arr(gamma, compare_data['gamma'])
        assert is_equal_arr(lve[2], compare_data['p'])
        assert is_equal_arr(lve[0], compare_data['x'])
        assert is_equal_arr(lve[1], compare_data['y'])

def gendata_binary():
    for comps in binaries:
        if comps in ('KR,AR', 'KR,C2'):
            continue
        comp1, comp2 = comps.split(',')
        print(comp1, comp2)
        data_out = {'T': [], 'gamma': [], 'x': [], 'y': [], 'p': []}
        for t in [0.7, 0.8, 0.9]:
            dft = PC_SAFT(','.join((comp1, comp2)))
            Tc = 1e6
            for z in [1e-3, 0.5, 1 - 1e-3]:
                Tc_new = dft.eos.critical([z, 1 - z])[0]
                print(f'Tc : {Tc_new}')
                if Tc_new < Tc:
                    Tc = Tc_new
            T = Tc * t
            print(f'T : {T}')

            gamma, lve = dft.surface_tension_isotherm(T, n_points=3, calc_lve=True, verbose=2)
            data_out['T'].extend([T for _ in gamma])
            data_out['gamma'].extend(gamma)
            data_out['x'].extend(lve[0])
            data_out['y'].extend(lve[1])
            data_out['p'].extend(lve[2])
        pd.DataFrame(data_out).to_csv(f'data/surf_tens_binary_{comp1}_{comp2}.csv')

def gendata():
    data_generators = [gendata_singlecomp, gendata_binary]
    for gen in data_generators:
        gen()

if __name__ == '__main__':
    gendata_binary()

    # gendata()