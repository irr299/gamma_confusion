from typing import Union

import numpy as np
from astropy.units import Quantity, Unit


class PowerLaw:
    def __init__(
        self,
        alpha: float,
        xmin: Quantity,
        xmax: Quantity,
    ) -> None:
        """Sample random variables from a power-law distribution.

        Parameters:
        -----------
        alpha : float
            Index of the power law.
        xmin : Quantity
            Minimum value.
        xmax : Quantity
            Maximum value.
        """

        self._alpha = alpha
        self._unit = xmin.unit
        self._xmin = xmin.value
        self._xmax = xmax.to(self._unit).value

    def get_sample(self, size: int, rng=np.random.default_rng()) -> Quantity:
        a = self._alpha + 1
        if not a == 0.0:
            r_range = self._xmax**a - self._xmin**a
            return (
                (rng.uniform(0, 1, size) * r_range + self._xmin**a) ** (1 / a)
            ).astype(np.float32) * self._unit
        else:
            return (
                10
                ** rng.uniform(
                    np.log10(np.array(self._xmin, dtype=np.float32)),
                    np.log10(np.array(self._xmax, dtype=np.float32)),
                    size,
                )
            ).astype(np.float32) * self._unit

    def __call__(self, size: int, rng=np.random.default_rng()) -> Quantity:
        return self.get_sample(size, rng)


class Exponential:
    def __init__(self, scale: float, unit: Union[str, Unit]) -> None:
        """Sample random variables from an exponential distribution.

        Parameters:
        -----------
        alpha : float
            The scale parameter.
        """
        self.scale = scale
        self.unit = Unit(unit)

    def __call__(self, size: int, rng=np.random.default_rng()) -> Quantity:
        return self.get_sample(size, rng)

    def get_sample(self, size: int, rng=np.random.default_rng()) -> Quantity:
        return rng.exponential(self.scale, size).astype(np.float32) * self.unit

    def __call__(self, size: int, rng=np.random.default_rng()) -> Quantity:
        return self.get_sample(size, rng)
