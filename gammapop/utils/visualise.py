from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.colors as cls
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from gammapy.maps import Map, WcsGeom


def _get_lon_center_and_width(
    coord_range: Union[Angle, Quantity], extent=1.0
) -> Tuple[Quantity, Quantity]:
    step = len(coord_range) // 3
    center = coord_range[step // 2 :: step]
    width = extent * np.diff(np.linspace(coord_range[0], coord_range[-1], 4))
    return center, width


def _get_lat_center_and_width(
    coord_range: Union[Angle, Quantity]
) -> Tuple[Quantity, Quantity]:
    lat_min = coord_range.min()
    lat_max = coord_range.max()
    return (lat_min + lat_max) / 2, lat_max - lat_min


def _get_split_geom(sky_map: Map) -> None:
    lon_center, lon_width = _get_lon_center_and_width(
        sky_map.geom.get_coord(mode="edges")["lon"][0, 0, :], extent=1.05
    )
    lon_width = np.fabs(lon_width.value)
    lat_center, lat_width = _get_lat_center_and_width(
        sky_map.geom.get_coord(mode="edges")["lat"][0, :, 0]
    )
    lat_width = np.fabs(lat_width.value)
    binsz = np.fabs(
        np.diff(sky_map.geom.get_coord(mode="edges")["lon"][0, 0, :])[0].value
    )
    geom1 = WcsGeom.create(
        skydir=(lon_center[0].value, lat_center.value),
        width=(lon_width[0], lat_width),
        binsz=binsz,
        proj="CAR",
        frame="galactic",
        axes=sky_map.geom.axes,
    )
    geom2 = WcsGeom.create(
        skydir=(lon_center[1].value, lat_center.value),
        width=(lon_width[1], lat_width),
        binsz=binsz,
        proj="CAR",
        frame="galactic",
        axes=sky_map.geom.axes,
    )
    geom3 = WcsGeom.create(
        skydir=(lon_center[2].value, lat_center.value),
        width=(lon_width[2], lat_width),
        binsz=binsz,
        proj="CAR",
        frame="galactic",
        axes=sky_map.geom.axes,
    )
    return geom1, geom2, geom3


def plot_sky_map(
    sky_map: Map, file_name: Optional[Union[Path, str]] = None, vmin: float = 1e-16
) -> None:
    geom1, geom2, geom3 = _get_split_geom(sky_map)

    fontsize = "xx-large"

    fig, axes = plt.subplots(
        3, 1, constrained_layout=True, sharex=False, figsize=(13.5, 7)
    )

    for index, (ax, g) in enumerate(zip(axes.ravel(), [geom1, geom2, geom3])):
        gd = sky_map.get_by_coord(g.get_coord())[0]
        plt.sca(ax)
        lonmin = g.get_coord()["lon"].to("deg").value[0, 0, 0]
        lonmax = g.get_coord()["lon"].to("deg").value[0, 0, -1]
        im = ax.imshow(
            gd,
            origin="lower",
            extent=[
                lonmin if lonmin < 180 else lonmin - 360,
                lonmax if lonmax < 180 else lonmax - 360,
                g.get_coord()["lat"].to("deg").value.min(),
                g.get_coord()["lat"].to("deg").value.max(),
            ],
            cmap="binary",
            norm=cls.LogNorm(
                vmin=max(vmin, np.nanmin(sky_map.data)), vmax=np.nanmax(sky_map.data)
            ),
        )
        ax.set_aspect("equal")
        plt.yticks(np.array([-3.0, 0.0, 3.0]), (r"$-3$", r"$0$", r"$3$"))
        if index == 2:
            ax.set_xlabel("Galactic Longitude [deg]", fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
    fig.text(
        -0.02,
        0.5,
        "Galactic Latitude [deg]",
        va="center",
        rotation="vertical",
        fontsize=fontsize,
    )

    fig.subplots_adjust(right=0.89)
    cbar_ax = fig.add_axes([1.0, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"Integral Flux [$cm^{-2}\,s^{-1}$]", fontsize=fontsize)

    cbar.ax.tick_params(labelsize=fontsize)

    if file_name:
        plt.savefig(file_name)
