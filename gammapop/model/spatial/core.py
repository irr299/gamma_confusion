from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.table import QTable
from astropy.units import Quantity
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import expon

from ..core import Model
from ..source.stellar import MainSequenceStar


class SymmetricDiscSpatialModel(Model):
    def __init__(
        self,
        r_offInKpc,
        R_sunInKpc,
        alpha,
        beta,
        z_scaleInKpc,
        rmaxInKpc=30,
        resolutionInKpc=0.01,
        galcen_distance: Optional[Quantity] = None,
        z_sun: Optional[Quantity] = None,
        fraction_run_away: float = 0.0,
    ):
        self.r_off = r_offInKpc
        self.R_sun = R_sunInKpc
        self.alpha = alpha
        self.beta = beta
        self.z_scale = z_scaleInKpc
        self.cdf_r_map = self.get_cdf_r_map(rmaxInKpc, resolutionInKpc)
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

    def get_r_density(self, r):
        return ((r + self.r_off) / (self.r_off + self.R_sun)) ** self.alpha * np.exp(
            -self.beta * (r - self.R_sun) / (self.r_off + self.R_sun)
        )

    def get_r_integrand(self, r):
        return r * self.get_r_density(r)

    def integrate_r_density(self, r):
        @np.vectorize
        def _integrate_r_density(r):
            return quad(self.get_r_integrand, 0, r)[0]

        return _integrate_r_density(r)

    def get_cdf_r_map(self, rmaxInKpc=30, resolutionInKpc=0.01):
        r_sample = np.linspace(0, rmaxInKpc, int(rmaxInKpc // resolutionInKpc) + 1)
        r_profile = self.integrate_r_density(r_sample[1:])
        r_profile = np.insert(r_profile, 0, 0)
        r_profile /= r_profile[-1]
        return interp1d(r_profile, r_sample)

    def get_random_r(self, size=1, rng=np.random.default_rng()):
        return self.cdf_r_map(rng.random(size)) * u.kpc

    @staticmethod
    def get_random_phi(size=1, rng=np.random.default_rng()):
        return 2 * np.pi * rng.random(size) * u.rad

    def get_random_z(self, size=1, rng=np.random.default_rng()):
        return (
            expon.rvs(scale=self.z_scale, size=size, random_state=rng)
            * (-1) ** rng.integers(0, 2, size=size)
            * u.kpc
        )

    def polar_to_cartesian(self, r, phi):
        x = r * np.cos(phi.to("rad").value)
        y = r * np.sin(phi.to("rad").value)
        return x, y

    def get_sample(self, size: int, rng=np.random.default_rng()) -> QTable:
        r = self.get_random_r(size, rng)
        phi = self.get_random_phi(size, rng)
        x, y = self.polar_to_cartesian(r, phi)
        z = self.get_random_z(size, rng)
        coord = SkyCoord(
            x=x.astype(np.float32),
            y=y.astype(np.float32),
            z=z.astype(np.float32),
            frame=self.frame,
        )
        if self.fraction_run_away > 0.0:
            coord = add_run_away(coord, self.fraction_run_away)
        return QTable({"coordinate": coord})


def add_run_away(coord: SkyCoord, fraction: float) -> SkyCoord:
    assert isinstance(
        coord.frame, Galactocentric
    ), "Need galactocentric coordinates to add run-away dispersion."
    assert 0.0 <= fraction <= 1.0, "fraction must lie in the interval [0,1]."
    n_run_away = int(np.round(len(coord) * fraction))
    stacked_coord = np.stack(
        [coord.x.to("kpc"), coord.y.to("kpc"), coord.z.to("kpc")], axis=-1
    )
    stacked_coord[
        np.random.choice(len(coord), n_run_away, replace=False)
    ] += _get_run_away_offsets(n_run_away)
    return SkyCoord(
        x=stacked_coord[:, 0],
        y=stacked_coord[:, 1],
        z=stacked_coord[:, 2],
        frame=coord.frame,
    )


def _get_run_away_offsets(size):
    sample = MainSequenceStar().get_sample(size)
    distance = (sample["velocity"] * sample["age"]).decompose().to("kpc")
    theta = np.pi * (np.random.rand(size) - 0.5)
    phi = 2 * np.pi * np.random.rand(size)
    z = distance * np.sin(theta)
    x = distance * np.cos(theta) * np.cos(phi)
    y = distance * np.cos(theta) * np.sin(phi)
    return np.stack([x, y, z], axis=-1)
