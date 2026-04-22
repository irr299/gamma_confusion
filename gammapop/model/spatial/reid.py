import warnings
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Optional, Union

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.table import QTable
from astropy.units import Quantity
from scipy.integrate import quad
from scipy.interpolate import interp1d

from ... import DEFAULT_RESOURCE_PATH
from ..core import Model
from .core import add_run_away


def _get_sample_height(
    theta, radius, flare_config, warp_config, rng=np.random.default_rng()
):
    loc, scale = _get_loc_and_scale_of_z(theta, radius, warp_config, flare_config)
    return rng.normal(loc, scale)


def _get_z_offset_with_warp(
    theta: np.ndarray,
    radius: np.ndarray,
    inner_radius: float,
    offset_angle: float,
    exponent: float,
    scale_height: float,
) -> np.ndarray:
    height = np.where(
        radius <= inner_radius, 0.0, scale_height * (radius - inner_radius) ** exponent
    )
    return height * np.sin(theta - offset_angle)


def _get_loc_and_scale_of_z(theta, radius, warp_config, flare_config):
    sigma = np.where(
        radius <= flare_config["inner_radius_in_kpc"],
        flare_config["sigma_off_in_kpc"],
        flare_config["sigma_off_in_kpc"]
        + flare_config["sigma_scale"] * (radius - flare_config["inner_radius_in_kpc"]),
    )

    theta_ = -theta + warp_config["sun_offset_in_radian"]
    loc = _get_z_offset_with_warp(
        theta_,
        radius,
        inner_radius=warp_config["inner_radius_in_kpc"],
        offset_angle=warp_config["offset_angle_in_radian"],
        exponent=warp_config["exponent"],
        scale_height=warp_config["scale_height_in_kpc"],
    )

    return loc, sigma


class InvalidSpiral(Exception):
    "Raised when the length of the spiral is zero"
    pass


class Spiral(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_theta(
        self,
    ) -> float:
        pass

    @staticmethod
    def _spiral_radius(theta, k, r_ref, theta_off):
        return r_ref * np.exp(k * (theta - theta_off))

    @staticmethod
    @np.vectorize
    def _get_r_density(r, r_off, r_sun, alpha, beta):
        return ((r + r_off) / (r_off + r_sun)) ** alpha * np.exp(
            -beta * (r - r_sun) / (r_off + r_sun)
        )

    def get_line_segment(
        self, theta_1, theta_2, k, r_ref, theta_off, r_off, r_sun, alpha, beta
    ):
        def integrand(theta):
            return np.sqrt(
                self._spiral_radius(theta, k, r_ref, theta_off) ** 2
                + (self._spiral_radius(theta, k, r_ref, theta_off) * k) ** 2
            ) * self._get_r_density(
                self._spiral_radius(theta, k, r_ref, theta_off),
                r_off,
                r_sun,
                alpha,
                beta,
            )

        return quad(integrand, theta_1, theta_2)[0]

    @abstractmethod
    def get_radius(self, size):
        pass

    @abstractmethod
    def get_sample(self, size, rng=np.random.default_rng()):
        pass


class LogSpiral(Spiral):
    def __init__(
        self,
        r_ref,
        tan_phi,
        theta_min,
        theta_max,
        below_kink=True,
        r_off=0.55,
        r_sun=8.5,
        alpha=1.64,
        beta=4.01,
        resolutionInKpc=0.01,
    ):
        theta_off = theta_max if below_kink else theta_min
        theta = np.array([theta_min, theta_max])
        r = r_ref * np.exp(tan_phi * (theta - theta_off))
        if r.min() >= r.max():
            raise InvalidSpiral("r_max must be greater than r_min")
        self.r_min = r.min()
        self.r_max = r.max()
        self.r_ref = r_ref
        self.tan_phi = tan_phi
        self.theta_off = theta_off
        self.weight = self.get_line_segment(
            theta_min, theta_max, tan_phi, r_ref, theta_off, r_off, r_sun, alpha, beta
        )
        self.cdf_r_map = self.get_cdf_r_map(r_off, r_sun, alpha, beta, resolutionInKpc)

    def get_theta(self, radius) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.log(radius / self.r_ref) / self.tan_phi + self.theta_off

    def get_r_integrand(self, r, r_off, r_sun, alpha, beta):
        return r * self._get_r_density(r, r_off, r_sun, alpha, beta)

    def integrate_r_density(self, r, r_off, r_sun, alpha, beta):
        @np.vectorize
        def _integrate_r_density(r):
            return quad(self.get_r_integrand, 0, r, args=(r_off, r_sun, alpha, beta))[0]

        return _integrate_r_density(r)

    def get_cdf_r_map(self, r_off, r_sun, alpha, beta, resolutionInKpc=0.01):
        r_sample = np.linspace(
            self.r_min,
            self.r_max,
            int(np.diff([self.r_min, self.r_max]) // resolutionInKpc) + 1,
        )
        r_profile = self.integrate_r_density(r_sample, r_off, r_sun, alpha, beta)
        r_profile -= r_profile[0]
        r_profile /= r_profile[-1]
        return interp1d(r_profile, r_sample)

    def get_radius(self, size, rng=np.random.default_rng()):
        return self.cdf_r_map(rng.random(size))

    def get_sample(self, size, rng=np.random.default_rng()):
        radius = self.get_radius(size, rng)
        theta = self.get_theta(radius)
        return radius, theta


class CircleSpiral(Spiral):
    def __init__(self, r_max) -> None:
        self.r_min = 0.0
        self.r_max = r_max
        self.weight = (
            2 * np.pi * r_max * self._get_r_density(r_max, 0.55, 8.5, 1.64, 4.01)
        )

    def get_theta(self, radius, rng=np.random.default_rng()) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return 2 * np.pi * rng.random(len(radius))

    def get_radius(self, size):
        return np.full(size, self.r_max)

    def get_sample(self, size, rng=np.random.default_rng()):
        radius = self.get_radius(size)
        theta = self.get_theta(radius, rng)
        return radius, theta


class ReidSpatialModel(Model):
    """Spiral arm model from Reid et al. 2019"""

    def __init__(
        self,
        resource_path: Optional[Union[str, Path]] = None,
        galcen_distance: Optional[Quantity] = None,
        z_sun: Optional[Quantity] = None,
        fraction_run_away: float = 0.0,
    ) -> None:
        file_name = "SpiralParametersCollection_Reid2019.yaml"
        file_path = (
            Path(resource_path, file_name)
            if resource_path
            else Path(DEFAULT_RESOURCE_PATH, file_name)
        )
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.spirals = self.get_spirals_from_config(config)
        self.weights = np.array([sp.weight for sp in self.spirals])
        self.weights = self.weights / self.weights.sum()
        self.r_max = np.max([sp.r_max for sp in self.spirals])
        self.get_width = partial(
            self._get_width,
            sigma_off=config["spiral_parameters"][0]["sigma_off_in_kpc"],
            sigma_scale=config["spiral_parameters"][0]["sigma_scale"],
            radius_off=config["spiral_parameters"][0]["inner_radius_in_kpc"],
        )
        self.get_height = partial(
            _get_sample_height,
            flare_config=config["flare_parameters"],
            warp_config=config["warp_parameters"],
        )
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

    def get_spirals_from_config(self, config: dict) -> list:
        spirals = []
        for spiral in config["spiral_parameters"]:
            if spiral["name"] == "3kpc":
                spirals += [CircleSpiral(spiral["radius_ref_in_kpc"])]
            else:
                try:
                    spirals += [
                        LogSpiral(
                            spiral["radius_ref_in_kpc"],
                            spiral["tan_phi"],
                            spiral["theta_min_in_radian"],
                            spiral["theta_kink_in_radian"],
                            below_kink=True,
                        )
                    ]
                except InvalidSpiral:
                    pass
                try:
                    spirals += [
                        LogSpiral(
                            spiral["radius_ref_in_kpc"],
                            spiral["tan_phi_kink"],
                            spiral["theta_kink_in_radian"],
                            spiral["theta_max_in_radian"],
                            below_kink=False,
                        )
                    ]
                except InvalidSpiral:
                    pass
        return spirals

    def get_theta(self, radius, rng=np.random.default_rng()) -> np.ndarray:
        theta = np.full_like(radius, np.nan)
        select = (radius <= self.r_max) & (radius >= 0.0)
        theta[select] = self._get_theta(radius[select], rng)
        return theta

    @staticmethod
    def _get_width(radius, sigma_off, sigma_scale, radius_off):
        return sigma_off + sigma_scale * (radius - radius_off)

    @staticmethod
    def _polar_to_cartesian(radius, theta):
        return radius * np.cos(theta), radius * np.sin(theta)

    @staticmethod
    def _cartesian_to_polar(x, y):
        return np.linalg.norm([x, y], axis=0), np.arctan2(y, x)

    def get_sample(self, size: int, rng=np.random.default_rng()) -> QTable:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spiral_index = rng.choice(
                np.arange(len(self.spirals)), size, replace=True, p=self.weights
            )
            samples = [
                self.spirals[iii].get_sample(np.sum(spiral_index == iii), rng)
                for iii in range(len(self.spirals))
            ]
            radius = np.concatenate([radius for radius, _ in samples])
            theta = np.concatenate([theta for _, theta in samples])
            x, y = self._polar_to_cartesian(radius, theta)
            delta_x, delta_y = self._polar_to_cartesian(
                np.fabs(rng.normal(scale=self.get_width(radius))),
                2 * np.pi * rng.random(size),
            )
            x += delta_x
            y += delta_y
            radius, theta = self._cartesian_to_polar(x, y)
            z = self.get_height(radius, theta, rng=rng)
            x, y = self._polar_to_cartesian(radius, theta + np.pi / 2)
        is_invalid = np.any(
            [
                np.isnan(x),
                np.isnan(y),
                np.isnan(z),
                np.isinf(x),
                np.isinf(y),
                np.isinf(z),
            ],
            axis=0,
        )
        coord = SkyCoord(
            x=x[~is_invalid].astype(np.float32) * u.kpc,
            y=y[~is_invalid].astype(np.float32) * u.kpc,
            z=z[~is_invalid].astype(np.float32) * u.kpc,
            frame=self.frame,
        )
        if self.fraction_run_away > 0.0:
            coord = add_run_away(coord, self.fraction_run_away)
        return QTable({"coordinate": coord})
