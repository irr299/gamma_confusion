import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.nddata import NoOverlapError
from astropy.table import Table
from astropy.units import Quantity
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import (
    GaussianSpatialModel,
    LogParabolaSpectralModel,
    PowerLaw2SpectralModel,
    SkyModel,
)
from tqdm import tqdm


def get_observation_window(
    energy_range: Quantity,
    lon_range: Angle,
    lat_range: Angle,
    resolution: Quantity = 0.1 * u.deg,
) -> WcsGeom:
    return WcsGeom.create(
        skydir=SkyCoord(l=lon_range.sum() / 2, b=lat_range.sum() / 2, frame="galactic"),
        width=(np.diff(lon_range), np.diff(lat_range)),
        binsz=resolution,
        frame="galactic",
        axes=[
            MapAxis(
                nodes=energy_range,
                interp="log",
                name="energy_true",
                node_type="edges",
            )
        ],
    )


def get_sky_model_powerlaw(
    coordinate: SkyCoord,
    flux: Quantity,
    extent: Quantity,
    emin: Quantity = 1 * u.TeV,
    emax=10 * u.TeV,
    name="source",
) -> SkyModel:
    coord = coordinate.transform_to("galactic")
    return SkyModel(
        spectral_model=PowerLaw2SpectralModel(
            index=2.3,
            amplitude=flux,
            emin=emin,
            emax=emax,
        ),
        spatial_model=GaussianSpatialModel(
            lon_0=coord.l,
            lat_0=coord.b,
            sigma=extent,
            frame="galactic",
        ),
        name=name,
    )


def get_sky_model_logparabola(
    coordinate: SkyCoord,
    amplitude: Quantity,
    e_ref: Quantity,
    alpha: float,
    beta: float,
    extent: Quantity,
    name="source",
) -> SkyModel:
    coord = coordinate.transform_to("galactic")
    return SkyModel(
        spectral_model=LogParabolaSpectralModel(
            amplitude=amplitude,
            reference=e_ref,
            alpha=alpha,
            beta=beta,
        ),
        spatial_model=GaussianSpatialModel(
            lon_0=coord.l,
            lat_0=coord.b,
            sigma=extent,
            frame="galactic",
        ),
        name=name,
    )


def get_sky_map(sources: Table, observation_window: WcsGeom) -> Map:
    data = np.zeros(observation_window.data_shape)
    if "beta" in sources.keys():
        params = (
            (co, am, re, al, be, ex)
            for co, am, re, al, be, ex in zip(
                sources["coordinate"],
                sources["amplitude"],
                sources["reference_energy"],
                sources["alpha"],
                sources["beta"],
                sources["extent"],
            )
        )
        model = get_sky_model_logparabola
    else:
        params = (
            (co, fl, ex)
            for co, fl, ex in zip(
                sources["coordinate"],
                sources["flux"],
                sources["extent"],
            )
        )
        model = get_sky_model_powerlaw
    for p in tqdm(params, total=len(sources)):
        sky_model = model(*p)
        try:
            flux = sky_model.integrate_geom(observation_window)
        except NoOverlapError:
            continue
        except ValueError as err:
            print(err)
            continue
        data += flux.data
    return Map.from_geom(
        geom=observation_window, data=data, unit=flux.unit
    ).sum_over_axes()
