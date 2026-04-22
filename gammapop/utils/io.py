from pathlib import Path
from typing import Union

import numpy as np
from astropy.coordinates import Galactic, Galactocentric, SkyCoord
from astropy.table import Table


def _split_skycoord(coordinate: SkyCoord) -> dict:
    if coordinate.is_equivalent_frame(Galactic()):
        return {
            "lon": coordinate.l.to("deg").astype(np.float32),
            "lat": coordinate.b.to("deg").astype(np.float32),
            "distance": coordinate.distance.to("kpc").astype(np.float32),
        }
    else:
        return {
            "x": coordinate.x.to("kpc").astype(np.float32),
            "y": coordinate.y.to("kpc").astype(np.float32),
            "z": coordinate.z.to("kpc").astype(np.float32),
            "z_sun": (coordinate.z_sun.to("pc") * np.ones(len(coordinate))).astype(
                np.float32
            ),
            "galcen_distance": (
                coordinate.galcen_distance.to("kpc") * np.ones(len(coordinate))
            ).astype(np.float32),
        }


def _convert_table_with_skycoord(table: Table) -> Table:
    return Table(
        {
            **{key: table[key] for key in table.keys() if key != "coordinate"},
            **_split_skycoord(table["coordinate"]),
        },
        meta=table.meta,
    )


def _merge_skycoord(table: Table) -> SkyCoord:
    if "lon" in table.keys():
        return SkyCoord(
            l=table["lon"].astype(np.float32),
            b=table["lat"].astype(np.float32),
            distance=table["distance"].astype(np.float32),
            frame="galactic",
        )
    else:
        return Galactocentric(
            x=table["x"].quantity.astype(np.float32),
            y=table["y"].quantity.astype(np.float32),
            z=table["z"].quantity.astype(np.float32),
            z_sun=table["z_sun"].quantity.astype(np.float32)[0],
            galcen_distance=table["galcen_distance"].quantity.astype(np.float32)[0],
        )


def _convert_table_with_coordinates(table: Table) -> Table:
    coord_keys = {"lon", "lat", "distance", "x", "y", "z", "z_sun", "galcen_distance"}
    return Table(
        {
            "coordinate": _merge_skycoord(table),
            **{key: table[key] for key in set(table.keys()) - coord_keys},
        },
        meta=table.meta,
    )


def write_table(
    table: Table, file_name: Union[Path, str], overwrite: bool = False
) -> None:
    _convert_table_with_skycoord(table).write(file_name, overwrite=overwrite)


def load_table(file_name: Union[Path, str]) -> Table:
    return _convert_table_with_coordinates(Table.read(file_name))
