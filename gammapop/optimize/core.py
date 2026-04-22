import numpy as np
from numba import njit, prange
from scipy.stats import poisson


@njit(fastmath=True)
def get_log10_flux(log10_L, log10_area_in_m2):
    return log10_L - log10_area_in_m2


@njit(fastmath=True)
def get_extent(radius_in_pc, distance_in_pc):
    return np.arctan(radius_in_pc / (distance_in_pc))


@njit(fastmath=True)
def get_log10_flux_threshold_for_extended_sources(
    log10_flux_in_m2s1_threshold, extent_in_rad, psf_in_rad
):
    return log10_flux_in_m2s1_threshold + 0.5 * np.log10(
        1 + (extent_in_rad / psf_in_rad) ** 2
    )


@njit(fastmath=True)
def is_extended_source_detectable(
    log10_flux_in_m2s1_threshold,
    log10_flux_in_m2s1,
    extent_in_rad,
    psf_in_rad,
    min_extent_in_rad,
    max_extent_in_rad,
):
    detectable = log10_flux_in_m2s1 >= get_log10_flux_threshold_for_extended_sources(
        log10_flux_in_m2s1_threshold, extent_in_rad, psf_in_rad
    )
    detectable &= extent_in_rad >= min_extent_in_rad
    detectable &= extent_in_rad <= max_extent_in_rad
    return detectable


@njit(fastmath=True)
def is_pointlike_source_detectable(
    log10_flux_in_m2s1_threshold,
    log10_flux_in_m2s1,
):
    return log10_flux_in_m2s1 >= log10_flux_in_m2s1_threshold


@njit(fastmath=True, parallel=True)
def get_fraction_of_detectable_extended_sources(
    log10_L_grid,
    log10_R_grid,
    log10_area_in_m2,
    distance_in_pc,
    log10_flux_in_m2s1_threshold,
    psf_in_rad,
    min_extent_in_rad,
    max_extent_in_rad,
):
    fraction = np.zeros_like(log10_L_grid)
    for ll in prange(log10_L_grid.shape[0]):
        for rr in prange(log10_L_grid.shape[1]):
            log10_flux_in_m2s1 = get_log10_flux(log10_L_grid[ll, rr], log10_area_in_m2)
            extent_in_rad = get_extent(10 ** log10_R_grid[ll, rr], distance_in_pc)
            detectable = is_extended_source_detectable(
                log10_flux_in_m2s1_threshold,
                log10_flux_in_m2s1,
                extent_in_rad,
                psf_in_rad,
                min_extent_in_rad,
                max_extent_in_rad,
            )
            fraction[ll, rr] += detectable.sum() / len(detectable)
    return fraction


@njit(fastmath=True, parallel=True)
def get_fraction_of_detectable_pointlike_sources(
    log10_L_grid,
    log10_area_in_m2,
    log10_flux_in_m2s1_threshold,
):
    fraction = np.zeros_like(log10_L_grid)
    for ll in prange(len(log10_L_grid)):
        log10_flux_in_m2s1 = get_log10_flux(log10_L_grid[ll], log10_area_in_m2)
        detectable = is_pointlike_source_detectable(
            log10_flux_in_m2s1_threshold,
            log10_flux_in_m2s1,
        )
        fraction[ll] += detectable.sum() / len(detectable)
    return fraction


def integrate_power_law(x, a):
    if a == -1.0:
        return np.log(x)
    else:
        return x ** (1 + a) / (1 + a)


def get_nll_1d(params, source_distribution, fraction, log10_L_bins):
    size, alpha_L = params
    L_bins = 10 ** (log10_L_bins - 34)
    prediction = np.diff(integrate_power_law(L_bins, alpha_L))
    prediction /= np.nansum(prediction)
    prediction *= size
    prediction *= fraction
    prediction[np.isnan(prediction)] = 0.0
    prediction[prediction < 0.0] = 0.0
    return -np.nansum(poisson.logpmf(source_distribution, mu=prediction))


def get_nll_2d(params, source_distribution, fraction, log10_L_bins, log10_R_bins):
    size, alpha_L, alpha_R = params
    L_bins = 10 ** (log10_L_bins - 34)
    R_bins = 10**log10_R_bins
    L_int = np.diff(integrate_power_law(L_bins, alpha_L))
    R_int = np.diff(integrate_power_law(R_bins, alpha_R))
    prediction = L_int[..., None] * R_int[None]
    prediction /= np.nansum(prediction)
    prediction *= size
    prediction *= fraction
    prediction[np.isnan(prediction)] = 0.0
    prediction[prediction < 0.0] = 0.0
    return -np.nansum(poisson.logpmf(source_distribution, mu=prediction))
