from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from astropy.table import QTable, hstack
from astropy.units import Quantity, Unit

from ..utils.convert import luminosity_to_flux, radius_to_extent


class Model(ABC):
    @abstractmethod
    def get_sample(self, size: int, rng=np.random.default_rng()) -> QTable:
        pass

    def __call__(self, size: int, rng=np.random.default_rng()) -> QTable:
        return self.get_sample(size, rng)


class PopulationModel(Model):
    def __init__(
        self,
        spatial_model: Model,
        source_model: Model,
        spectrum_model: Optional[Model] = None,
    ) -> None:
        self.spatial_model = spatial_model
        self.source_model = source_model
        self.spectrum_model = spectrum_model

    def get_sample(self, size: int, rng=np.random.default_rng()) -> QTable:
        coord = self.spatial_model(size, rng)["coordinate"].transform_to("galactic")
        properties = self.source_model(size, rng)
        table = QTable(
            {
                "coordinate": coord,
                "luminosity": properties["luminosity"],
                "radius": properties["radius"],
                "flux": luminosity_to_flux(
                    properties["luminosity"], coord.distance
                ).astype(np.float32),
                "extent": radius_to_extent(properties["radius"], coord.distance).astype(
                    np.float32
                ),
            },
            meta={
                "SUN_X": np.round(
                    -self.spatial_model.frame.galcen_distance.to_value("kpc"), 4
                ),
                "SUN_Y": 0.0,
                "SUN_Z": np.round(self.spatial_model.frame.z_sun.to_value("kpc"), 4),
                "SUN_UNIT": "kpc",
            },
        )
        if self.spectrum_model:
            table = hstack([table, self.spectrum_model.get_sample(table["flux"], rng)])
        return table
