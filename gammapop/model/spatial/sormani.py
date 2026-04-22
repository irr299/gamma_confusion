from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.table import QTable, Table
from astropy.units import Quantity

from ... import DEFAULT_RESOURCE_PATH
from ..core import Model
from .core import add_run_away


class SormaniSpatialModel(Model):
    """Bar/bulge model from Sormani et al. 2022"""

    def __init__(
        self,
        resource_path: Optional[Union[str, Path]] = None,
        galcen_distance: Optional[Quantity] = None,
        z_sun: Optional[Quantity] = None,
        fraction_run_away: float = 0.0,
    ) -> None:
        file_name = "bar_sormani.fits"
        file_path = (
            Path(resource_path, file_name)
            if resource_path
            else Path(DEFAULT_RESOURCE_PATH, file_name)
        )
        self.table = Table.read(file_path)
        self.fraction_run_away = fraction_run_away

        default_frame = Galactocentric()
        gc_dist = (
            galcen_distance
            if isinstance(galcen_distance, Quantity)
            else default_frame.galcen_distance
        )
        zsun = z_sun if isinstance(z_sun, Quantity) else default_frame.z_sun
        self.frame = Galactocentric(
            galcen_distance=gc_dist,
            z_sun=zsun,
        )

    def get_sample(self, size, rng=np.random.default_rng()) -> QTable:
        sample = self.table[rng.choice(len(self.table), size, p=self.table["density"])]
        coord = SkyCoord(
            x=sample["x"].quantity,
            y=sample["y"].quantity,
            z=sample["z"].quantity,
            frame=self.frame,
        )
        if self.fraction_run_away > 0.0:
            coord = add_run_away(coord, self.fraction_run_away)
        return QTable({"coordinate": coord})
