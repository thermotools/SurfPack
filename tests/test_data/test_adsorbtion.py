from surfpack.pcsaft import PC_SAFT
import pandas as pd
import pytest
from tests.tools import is_equal, is_equal_arr, singlecomps, singlecomps2, binaries

@pytest.mark.parametrize('comps', binaries)
@pytest.mark.parametrize('t', [0.7, 0.8, 0.9])
def test_binary(comps, z, t):
    comp1, comp2 = comps.split(',')
    compare_data = pd.read_csv(f'data/adsorbtion_binary_{comp1}_{comp2}.csv')
    dft = PC_SAFT(','.join((comp1, comp2)))
    Tc = 1e6
    for z in [1e-3, 0.5, 1 - 1e-3]:
        Tc_new = dft.eos.critical([z, 1 - z])[0]
        if Tc_new < Tc:
            Tc = Tc_new
    T = Tc * t
    compare_data = compare_data[compare_data['T'] == T]
    ads, lve = dft.adsorbtion_isotherm(T, n_points=3, x_min=0.2, x_max=0.8, calc_lve=True)
    assert is_equal_arr(ads[0], compare_data['gamma (1)'])
    assert is_equal_arr(ads[1], compare_data['gamma (2)'])
    assert is_equal_arr(lve[2], compare_data['p'])
    assert is_equal_arr(lve[0], compare_data['x'])
    assert is_equal_arr(lve[1], compare_data['y'])


def gendata_binary():
    for comps in binaries:
        print(comps)
        comp1, comp2 = comps.split(',')
        data_out = {'T': [], 'gamma (1)': [], 'gamma (2)': [], 'x': [], 'y': [], 'p': []}
        for t in [0.7, 0.8, 0.9]:
            dft = PC_SAFT(','.join((comp1, comp2)))
            Tc = 1e6
            for z in [1e-3, 0.5, 1 - 1e-3]:
                Tc_new = dft.eos.critical([z, 1 - z])[0]
                if Tc_new < Tc:
                    Tc = Tc_new
            T = Tc * t
            print(T, Tc)
            ads, lve = dft.adsorbtion_isotherm(T, n_points=3, x_min=0.2, x_max=0.8, calc_lve=True, verbose=2)
            data_out['T'].extend([T for _ in ads[0]])
            data_out['gamma (1)'].extend(ads[0])
            data_out['gamma (2)'].extend(ads[1])
            data_out['x'].extend(lve[0])
            data_out['y'].extend(lve[1])
            data_out['p'].extend(lve[2])
        pd.DataFrame(data_out).to_csv(f'data/adsorbtion_binary_{comp1}_{comp2}.csv')


def gendata():
    data_generators = [gendata_binary]
    for gen in data_generators:
        gen()


if __name__ == '__main__':
    gendata()