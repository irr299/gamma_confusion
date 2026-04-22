import numpy as np
import astropy.units as u
from astropy.units import Quantity


def verify_is_quantity(x):
    assert isinstance(x, Quantity), f"Expected a Quantity object but got {type(x)}."


def luminosity_to_flux(luminosity: Quantity, distance: Quantity) -> Quantity:
    verify_is_quantity(luminosity)
    verify_is_quantity(distance)
    return (luminosity / (4 * np.pi * distance**2)).to(luminosity.unit / u.cm**2)


def flux_to_luminosity(flux: Quantity, distance: Quantity) -> Quantity:
    verify_is_quantity(flux)
    verify_is_quantity(distance)
    return (flux * 4 * np.pi * distance**2).decompose()


def radius_to_extent(radius: Quantity, distance: Quantity) -> Quantity:
    verify_is_quantity(radius)
    verify_is_quantity(distance)
    return np.arctan(radius / distance).to("deg")


def extent_to_radius(extent: Quantity, distance: Quantity) -> Quantity:
    verify_is_quantity(extent)
    verify_is_quantity(distance)
    return (np.tan(extent) * distance).to("pc")
