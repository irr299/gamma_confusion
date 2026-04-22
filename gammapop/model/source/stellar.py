from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.constants import M_sun
from astropy.table import QTable
from astropy.units import Quantity
from scipy.interpolate import interp1d

from ... import DEFAULT_RESOURCE_PATH
from ..utils import Exponential, PowerLaw


class MainSequenceStar:
    def __init__(
        self,
    ) -> None:
        data = np.genfromtxt(
            Path(DEFAULT_RESOURCE_PATH, "data_massive_progenitors.txt")
        )
        self._mass_to_age = interp1d(
            np.log(data[:, 0]), np.log(data[:, 2]), kind="quadratic"
        )
        self.initial_mass_function = PowerLaw(
            -2.3, data[0, 0] * M_sun, data[-1, 0] * M_sun
        )
        self.velocity_scale = Exponential(scale=150, unit="km s-1")

    def get_mass(self, size: int, rng=np.random.default_rng()) -> Quantity:
        return self.initial_mass_function(size, rng)

    def get_velocity(self, size: int, rng=np.random.default_rng()) -> Quantity:
        return self.velocity_scale(size, rng)

    def mass_to_age(self, mass: Quantity) -> Quantity:
        return (
            np.exp(self._mass_to_age(np.log((mass / M_sun).decompose()))).astype(
                np.float32
            )
            * u.yr
        )

    def get_sample(self, size: int, rng=np.random.default_rng()) -> QTable:
        mass = self.get_mass(size, rng)
        velocity = self.get_velocity(size, rng)
        age = self.mass_to_age(mass)
        return QTable(
            {
                "mass": mass,
                "velocity": velocity,
                "age": age,
            }
        )
