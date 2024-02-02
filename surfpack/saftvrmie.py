from surfpack.saft import SAFT
from surfpack.saft_hardsphere import SAFT_WhiteBear
from thermopack.saftvrmie import saftvrmie

class SAFT_VR_Mie(SAFT):
    def __init__(self, comps, hs_model=SAFT_WhiteBear):
        super().__init__(comps, saftvrmie, hs_model)

    def get_characteristic_lengths(self):
        return self.eos.get_pure_fluid_param(1)[1] * 1e10