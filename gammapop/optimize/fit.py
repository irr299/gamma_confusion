import warnings
from functools import partial
from typing import Optional, Union

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.optimize import minimize
from scipy.stats import norm, uniform
from tqdm import tqdm

from ..survey.core import Survey
from ..utils.convert import extent_to_radius, flux_to_luminosity
from .core import (
    get_fraction_of_detectable_extended_sources,
    get_fraction_of_detectable_pointlike_sources,
    get_nll_1d,
    get_nll_2d,
)


def get_fraction_of_detectable_sources(
    coordinates: SkyCoord,
    survey: Survey,
    log10_L_bins: np.ndarray,
    log10_R_bins: Optional[np.ndarray] = None,
) -> np.ndarray:
    gal_coord = coordinates.transform_to("galactic")
    gal_coord_no_distance = SkyCoord(l=gal_coord.l, b=gal_coord.b, frame="galactic")
    log10_area_in_m2 = np.log10(4 * np.pi * (gal_coord.distance**2).to("m2").value)
    log10_flux_in_m2s1_threshold = np.log10(
        survey.get_detection_threshold_for_point_sources(gal_coord_no_distance)
        .decompose()
        .value
    )
    log10_L_points = log10_L_bins[:-1] + np.diff(log10_L_bins) / 2
    if not log10_R_bins is None:
        distance_in_pc = gal_coord.distance.to("pc").value
        log10_R_points = log10_R_bins[:-1] + np.diff(log10_R_bins) / 2
        grid = np.meshgrid(log10_L_points, log10_R_points, indexing="ij")
        get_fraction = partial(
            get_fraction_of_detectable_extended_sources,
            log10_area_in_m2=log10_area_in_m2,
            distance_in_pc=distance_in_pc,
            log10_flux_in_m2s1_threshold=log10_flux_in_m2s1_threshold,
            psf_in_rad=survey.psf.to(u.rad).value,
            min_extent_in_rad=survey.min_extent.to(u.rad).value,
            max_extent_in_rad=survey.max_extent.to(u.rad),
        )
        # dry run to compile numba function
        _ = get_fraction(np.ones(4).reshape(2, 2), np.ones(4).reshape(2, 2))
    else:
        grid = tuple([log10_L_points])
        get_fraction = partial(
            get_fraction_of_detectable_pointlike_sources,
            log10_area_in_m2=log10_area_in_m2,
            log10_flux_in_m2s1_threshold=log10_flux_in_m2s1_threshold,
        )
        # dry run to compile numba function
        _ = get_fraction(np.ones(2))
    # actual computation
    fraction = get_fraction(*grid)
    return fraction, grid


def _is_fit_result_valid(result):
    return (
        result.success
        or result.message
        == "Desired error not necessarily achieved due to precision loss."
    )


def get_optimised_parameters(
    sources: Table,
    fraction: np.ndarray,
    log10_L_bins: np.ndarray,
    log10_R_bins: Optional[np.ndarray] = None,
    n_iter: int = 1_000,
    x0: Optional[Union[np.ndarray, list]] = None,
):
    flux = norm(loc=sources["flux"], scale=sources["flux_err"])
    flux_unit = sources["flux"].unit
    try:
        distance = uniform(loc=sources["distance_min"], scale=sources["distance_max"])
        distance_unit = sources["distance_min"].unit
    except KeyError:
        distance = norm(loc=sources["distance"], scale=1e-16)
        distance_unit = sources["distance"].unit

    if not log10_R_bins is None:
        extent = norm(loc=sources["extent"], scale=sources["extent_err"])
        extent_unit = sources["extent"].unit

        def _get_distribution():
            distance_rvs = distance.rvs() * distance_unit
            luminosity = flux_to_luminosity(flux.rvs() * flux_unit, distance_rvs)
            radius = extent_to_radius(extent.rvs() * extent_unit, distance_rvs)
            return np.histogram2d(
                np.log10(luminosity.to("s-1").value),
                np.log10(radius.to("pc").value),
                bins=[log10_L_bins, log10_R_bins],
            )[0]

        get_nll = partial(
            get_nll_2d,
            fraction=fraction,
            log10_L_bins=log10_L_bins,
            log10_R_bins=log10_R_bins,
        )
        get_fit_result = partial(
            minimize, fun=get_nll, x0=[100, 1, 1] if x0 is None else x0
        )

    else:

        def _get_distribution():
            luminosity = flux_to_luminosity(
                flux.rvs() * flux_unit, distance.rvs() * distance_unit
            )
            return np.histogram(
                np.log10(luminosity.to("s-1").value),
                bins=log10_L_bins,
            )[0]

        get_nll = partial(get_nll_1d, fraction=fraction, log10_L_bins=log10_L_bins)
        get_fit_result = partial(
            minimize, fun=get_nll, x0=[100, 1] if x0 is None else x0
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source_distribution = (_get_distribution() for _ in range(n_iter))
        fit_results = (
            get_fit_result(args=dist)
            for dist in tqdm(source_distribution, total=n_iter)
        )
        stacked_result = np.stack(
            [res.x for res in fit_results if _is_fit_result_valid(res)], axis=0
        ).astype(np.float32)
    table = Table({"size": stacked_result[:, 0], "alpha_L": stacked_result[:, 1]})
    if not log10_R_bins is None:
        table["alpha_R"] = stacked_result[:, 2]

    return table
