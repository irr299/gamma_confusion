from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.table import QTable
from astropy.units import Quantity
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm

from ..core import Model
from .core import add_run_away


class SteimanSpatialModel(Model):
    """Spiral arm model from Steiman-Cameron et al. 2010"""

    def __init__(
        self,
        rmaxInKpc=30,
        resolutionInKpc=0.01,
        galcen_distance: Optional[Quantity] = None,
        z_sun: Optional[Quantity] = None,
        fraction_run_away: float = 0.0,
    ):
        self.r_peak = 2.9  # kpc
        self.r_scale_min = 0.7  # kpc
        self.r_scale_max = 3.1  # kpc
        self.z_scale = 0.07  # kpc
        self.spiral_scale = np.radians(15)  # rad
        # spiral parameters (a [kpc], beta [rad-1]]
        self.spiral_params = np.array(
            [
                [0.246, 0.242],  # Sagittarius-Carina
                [0.608, 0.279],  # Scutum-Crux
                [0.449, 0.249],  # Perseus
                [0.378, 0.240],  # Norma-Cygnus
            ]
        )
        self.spiral_weights = np.array(
            [
                169,  # Sagittarius-Carina
                266,  # Scutum-Crux
                339,  # Perseus
                176,  # Norma-Cygnus
            ],
            dtype=np.float32,
        )
        self.spiral_weights /= np.sum(self.spiral_weights)
        self.cdf_radius_map = self.get_cdf_r_map(rmaxInKpc, resolutionInKpc)
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
        x = r - self.r_peak
        scale = np.where(x < 0, self.r_scale_min, self.r_scale_max)
        return np.exp(-np.fabs(x) / scale)

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

    @staticmethod
    def get_phi(rInKpc, a, beta):
        return np.log(rInKpc / a) / beta * u.rad

    def get_random_r(self, size=1, rng=np.random.default_rng()):
        return self.cdf_radius_map(rng.random(size)) * u.kpc

    def get_random_phi(self, r, a, beta, rng=np.random.default_rng()):
        phi = self.get_phi(r, a, beta)
        return (
            norm.rvs(
                size=len(phi),
                loc=phi.to("rad").value,
                scale=self.spiral_scale,
                random_state=rng,
            )
            * u.rad
        )

    def get_random_z(self, size=1, rng=np.random.default_rng()):
        return (
            norm.rvs(size=size, loc=0.0, scale=self.z_scale, random_state=rng) * u.kpc
        )

    def get_random_arm(self, size=1, rng=np.random.default_rng()):
        return rng.choice(
            np.arange(len(self.spiral_weights)),
            size=size,
            replace=True,
            p=self.spiral_weights,
        )

    def polar_to_cartesian(self, r, phi):
        """In Steiman-Cameron et al. the coordinate system is defined such that
        the Sun is located at (x=0.0 pc, y=8.5 pc). Here, we rotate the coordinates
        by +90 deg such that the source positions comply with the astropy definition
        of the Galactocentric coordinate system, which places the Sun at
        ~(x=-8.0 pc, y=0.0 pc)."""
        x = r * np.cos(phi.to("rad").value + np.pi / 2)
        y = r * np.sin(phi.to("rad").value + np.pi / 2)
        return x, y

    def get_sample(self, size: int, rng=np.random.default_rng()) -> QTable:
        r = self.get_random_r(size, rng)
        arm = self.get_random_arm(size, rng)
        spiral_params = self.spiral_params[arm]
        phi = self.get_random_phi(
            r.to("kpc").value, spiral_params[:, 0], spiral_params[:, 1], rng
        )
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
