from typing import Optional, Union

import numpy as np
from astropy import units
from astropy.table import QTable, hstack
from scipy.stats import multivariate_normal

from ..core import Model
from ..utils import PowerLaw


class EmptyValueModel(Model):
    def __init__(self, quantity_name: str, unit: Union[str, units.Unit]) -> None:
        """Filler for unused quantities which returns NaN values.

        Parameters:
        -----------
        quantity_name : str
            _description_.
        unit : Union[str, units.Unit]
            _description_.
        """

        super().__init__()
        self.name = quantity_name
        self.unit = unit

    def get_sample(self, size: int, rng=None) -> QTable:
        return QTable(
            {self.name: np.full(size, np.nan, dtype=np.float32) * units.Unit(self.unit)}
        )


class PowerLawModel(Model):
    def __init__(
        self,
        quantity_name: str,
        alpha: float,
        xmin: units.Quantity,
        xmax: units.Quantity,
    ) -> None:
        """Sample values from a power-law distribution.

        Parameters:
        -----------
        quantity_name : str
            Name of the quantity to sample.
        alpha : float
            Index of the power law.
        xmin : units.Quantity
            Minimum value of the quantity.
        xmax : units.Quantity
            Maximum value of the quantity.
        """

        super().__init__()
        self.name = quantity_name
        self.model = PowerLaw(alpha, xmin, xmax)

    def get_sample(self, size: int, rng=np.random.default_rng()) -> QTable:
        return QTable({self.name: self.model.get_sample(size, rng)})


class IndependentSourcePropertyModel(Model):
    def __init__(
        self,
        luminosity_model: Model,
        radius_model: Model,
    ) -> None:
        super().__init__()
        self.luminosity_model = luminosity_model
        self.radius_model = radius_model

    def get_sample(self, size: int, rng=np.random.default_rng()) -> QTable:
        return hstack(
            [
                self.luminosity_model.get_sample(size, rng),
                self.radius_model.get_sample(size, rng),
            ]
        )


class LBNSourcePropertyModel(Model):
    def __init__(
        self,
        m1: float,
        m2: float,
        s1: float,
        s2: float,
        cor: float,
        seed: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
    ) -> None:
        """Samples source luminosities and radii from a log-bivariate Normal
        (i.e. a probability distribution over log_10(L) and log_10(R)).

        See e.g.: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

        Parameters:
        -----------
        m1 : float
            Mean of the marginal log_10(L / (1 photons/s)) distribution.
        m2 : float
            Mean of the marginal log_10(R / (1 pc)) distribution.
        s1 : float
            Standard deviation of the marginal log_10(L / (1 photons/s)) distribution.
        s2 : float
            Standard deviation of the marginal log_10(R / (1 pc)) distribution.
        cor : float
            Correlation parameter. Must be in the interval [-1, 1].
        """
        super().__init__()

        assert -1.0 <= cor <= 1.0, "The correlation must lie in the interval [-1,1]."
        mean = np.array([m1, m2])
        covariance = np.array(
            [
                [s1**2, cor * s1 * s2],
                [cor * s1 * s2, s2**2],
            ]
        )
        self._mvn = multivariate_normal(
            mean, covariance, allow_singular=True, seed=seed
        )

    def get_sample(self, size: int, rng=np.random.default_rng()) -> QTable:
        sample = self._mvn.rvs(size=size, random_state=rng)
        return QTable(
            {
                "luminosity": (10 ** sample[:, 0]) / units.s,
                "radius": (10 ** sample[:, 1]).astype(np.float32) * units.pc,
            }
        )
