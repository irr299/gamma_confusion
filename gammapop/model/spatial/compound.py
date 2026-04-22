import numpy as np
from astropy.table import QTable, vstack

from ..core import Model
from .reid import ReidSpatialModel
from .sormani import SormaniSpatialModel


class CompoundSpatialModel(Model):
    """A model that combines a disc model with a bar/bulge component."""

    def __init__(
        self,
        disc: Model = ReidSpatialModel(),
        bar: Model = SormaniSpatialModel(),
        fraction_bar: int = 0.35,
    ) -> None:
        self.disc = disc
        self.bar = bar
        self.fraction_bar = fraction_bar
        self.frame = disc.frame

    def get_sample(self, size, rng=np.random.default_rng()) -> QTable:
        disc_size = int(np.round(size / (1 + self.fraction_bar)))
        bar_size = int(np.round(disc_size * self.fraction_bar))
        disc_size += size - (disc_size + bar_size)
        return vstack(
            [
                self.disc(disc_size, rng),
                self.bar(bar_size, rng),
            ]
        )
