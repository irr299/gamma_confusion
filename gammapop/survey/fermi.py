import logging
import warnings
from pathlib import Path
from urllib import request

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import QTable, join
from gammapy.maps import Map

from .. import DEFAULT_RESOURCE_PATH
from .core import Survey

logger = logging.getLogger(__name__)


class FGL(Survey):
    """A class to represent the 4FGL catalog.

    Details can be found here: https://fermi.gsfc.nasa.gov/ssc/data/access/lat/12yr_catalog/

    Attributes
    ----------
    psf : Quantity
        The average angular resolution across the FoV (68% containment
        radius for a point source).
    min_extent : Quantity
        Minimum angular extent of a source to be recognized as being
        extended. Value corresponds to
        the estimated systematic error on the PSF.
    max_extent : Quantity
        Maximum angular extent of a source to be detectable.
    source_table : Table
        Table of 4FGL sources with their properties.
    he_source_table : Table
        Table of high energy Fermi sources.
    sensitivity_map : Map
        Sensitivity of the 4FGL to point sources across the FoV.
    """

    _resource_url = "https://fermi.gsfc.nasa.gov/ssc/data/access/lat/12yr_catalog/"

    def __init__(self):
        table_file_name = "gll_psc_v31.fit"
        sensitivity_map_file_name = "detthresh_P8R3_12years_PL22.fits"
        file_path_table = self._get_path_to_file(table_file_name)
        file_path_sensitivity_map = self._get_path_to_file(sensitivity_map_file_name)
        self.psf = (
            0.4 * u.deg
        )  # average value across the field of view (guess from fig1)
        self.min_extent = 0.08 * u.deg  # size of the smallest extended source
        self.max_extent = 3.45 * u.deg  # size of the largest source in the catalogue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgl_table = QTable.read(file_path_table, hdu=1)
            fgl_distances = ascii.read(
                Path(DEFAULT_RESOURCE_PATH, "fgl_distances.CSV"),
                format="csv",
                fast_reader=False,
                names=["ASSOC1", "Distance"],
                delimiter=";",
            )
            fgl_distances["Distance"] = fgl_distances["Distance"] * u.deg
            fgl_distances["ASSOC1"] = np.array(
                list(map(lambda x: f"{x:<28}", fgl_distances["ASSOC1"])), dtype="<U28"
            )
            fgl_table = join(fgl_table, fgl_distances, join_type="left", keys="ASSOC1")
            fgl_table["Distance"][fgl_table["Distance"].mask] = np.nan

            # Add alias
            self.source_table = QTable(
                [
                    fgl_table["Source_Name"],
                    fgl_table["Energy_Flux100"].astype(np.float32),
                    np.zeros(len(fgl_table)).astype(np.float32),
                    SkyCoord(
                        l=fgl_table["GLON"].astype(np.float32),
                        b=fgl_table["GLAT"].astype(np.float32),
                        distance=fgl_table["Distance"].unmasked,
                        frame="galactic",
                    ),
                    fgl_table["Distance"].astype(np.float32),
                ],
                names=("Source_Name", "flux", "extent", "coordinate", "distance"),
            )

            self.sensitivity_map = Map.read(file_path_sensitivity_map)
            self.sensitivity_map.data = np.where(
                np.greater(self.sensitivity_map.data, 0.0),
                self.sensitivity_map.data,
                np.nan,
            )

    def _get_path_to_file(self, file_name: str) -> Path:
        """Return the path to the file in the default directory.

        Check if the file is already available and download it if it is not.

        Parameters
        ----------
        file_name : str
            The name of the file.

        Returns
        -------
        Path
            Path to file.
        """

        file_path = Path(DEFAULT_RESOURCE_PATH, file_name)
        if not file_path.is_file():
            logger.info(f"Downloading '{file_name}'")
            request.urlretrieve(self._resource_url + file_name, file_path)
            logger.info(f"Downloaded '{file_name}'")
        return file_path

    def get_detection_threshold_for_point_sources(
        self, sky_coord: SkyCoord
    ) -> u.Quantity:
        """Yields the minimum flux required for the detection of a point source
        at the given coordinates.

        Parameters
        ----------
        sky_coord : SkyCoord
            The position of the source in celestial coordinates.

        Returns
        -------
        Quantity
            The minimum integral flux >1 TeV.
        """

        map_values = self.sensitivity_map.get_by_coord(sky_coord)
        # using 'not greater' handles NaN values appropriately unlike '<='
        outside_analysis_region = ~np.greater(map_values, 0.0)
        map_values[outside_analysis_region] = np.inf
        return map_values * self.sensitivity_map.unit

    def get_detection_threshold(self, sky_coord: SkyCoord) -> u.Quantity:
        return self.get_detection_threshold_for_point_sources(sky_coord)

    def is_visible(self, sky_coord: SkyCoord) -> np.ndarray:
        """Determines the visibility of a source within the HGPS (i.e. source
        appears inside FoV).

        Parameters
        ----------
        sky_coord : SkyCoord
            The position of the source in celestial coordinates.

        Returns
        -------
        bool
            The visibility.
        """

        return ~np.isinf(self.get_detection_threshold(sky_coord))

    def is_detectable(self, sky_coord: SkyCoord, flux: u.Quantity) -> np.ndarray:
        """Determines the detectability of a source within the 4FGL.

        Parameters
        ----------
        sky_coord : SkyCoord
            The position of the source in celestial coordinates.
        flux : Quantity
            The integral flux between 100 MeV and 1 TeV.

        Returns
        -------
        bool
            The detectability.
        """

        return np.greater_equal(flux, self.get_detection_threshold(sky_coord))
