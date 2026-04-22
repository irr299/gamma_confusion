from abc import ABC, abstractmethod

import astropy.units as u

INTEGRAL_PHOTON_FLUX_UNIT = u.cm**-2 * u.s**-1
DIFFERENTIAL_PHOTON_FLUX_UNIT = u.cm**-2 * u.s**-1 * u.TeV**-1
INTEGRAL_FLUX_ABOVE_1_TEV_CRAB = 2.66e-11 * INTEGRAL_PHOTON_FLUX_UNIT


class Survey(ABC):
    @abstractmethod
    def is_visible(self):
        pass

    @abstractmethod
    def is_detectable(self):
        pass
