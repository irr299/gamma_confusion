"""This module provides an interface source catalogues derived with H.E.S.S.

Author: Constantin Steppa (2023)
"""

import logging
import warnings
from pathlib import Path
from urllib import request

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, join
from astropy.table.table import MaskedColumn, QTable, Table
from astropy.units import Quantity
from gammapy.maps import Map
from regions import RectangleSkyRegion
from scipy.stats import halfnorm

from .. import DEFAULT_RESOURCE_PATH
from .core import DIFFERENTIAL_PHOTON_FLUX_UNIT, INTEGRAL_PHOTON_FLUX_UNIT, Survey

logger = logging.getLogger(__name__)


class HGPS(Survey):
    """A class to represent the H.E.S.S. Galactic plane survey.

    Details of the HGPS are described in: https://doi.org/10.1051/0004-6361/201732098 .
    The data being used was downloaded from: https://www.mpi-hd.mpg.de/hfm/HESS/hgps/ .

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
    galactic_center_cutout_region : RectangleSkyRegion
        The region around the Galactic center that was excluded from
        the analysis of the HGPS data.
    source_table : Table
        Table of HGPS sources with their properties.
    sensitivity_map : Map
        Sensitivity of the HGPS to point sources across the FoV.
    """

    _resource_url = "https://www.mpi-hd.mpg.de/hfm/HESS/hgps/data/"

    def __init__(
        self,
        bin_size_in_deg: str = "0.2",
        sensitivity_scale: float = 1.0,
        detection_psf: u.Quantity | None = None,
    ) -> None:
        """
        Parameters
        ----------
        bin_size_in_deg : str
            The resolution of the sensisitivity map. The map is available
            with 0.1 deg and 0.2 deg resolution (default '0.2'). Note
            that this method will fall back to the default value if
            provided with an invalid parameter!
        sensitivity_scale : float
            Multiplicative factor applied to the sensitivity map threshold.
            Values < 1 lower the detection bar (more detections); values > 1
            raise it. Default is 1.0 (no change, matching published HGPS).
        detection_psf : Quantity or None
            Effective PSF used *only* in the extended-source threshold
            correction: ``sqrt(1 + (extent / detection_psf)^2)``.
            When None (default), falls back to ``self.psf`` (0.08 deg).
            Increasing this value softens the extent penalty.
        """

        logger.info("Creating an instance of HGPS")
        valid_bin_sizes = ["0.1", "0.2"]
        try:
            assert (
                bin_size_in_deg in valid_bin_sizes
            ), f"Argument 'bin_size_in_deg' must be one of {valid_bin_sizes}!"
            _bin_size_in_deg = bin_size_in_deg
        except AssertionError:
            logger.exception(
                f"An invalid value for 'bin_size_in_deg' was given ({bin_size_in_deg}). "
                "Continuing with default value ('0.2')."
            )
            _bin_size_in_deg = "0.2"
        self.psf = 0.08 * u.deg  # average value across the field of view
        self.min_extent = (
            0.03 * u.deg
        )  # systematic bias on the size of point-like sources
        self.max_extent = 1.0 * u.deg  # size of the largest source in the catalogue
        self.galactic_center_cutout_region = RectangleSkyRegion(
            center=SkyCoord(l=0.25 * u.deg, b=0.0 * u.deg, frame="galactic"),
            width=2.5 * u.deg,
            height=1.0 * u.deg,
        )
        colnames1 = [
            "Source_Name",
            "Analysis_Reference",
            "Source_Class",
            "Identified_Object",
            "GLON",
            "GLON_Err",
            "GLAT",
            "GLAT_Err",
            "Pos_Err_68",
            "Pos_Err_95",
            "Spatial_Model",
            "Components",
            "Sqrt_TS",
            "Size",
            "Size_Err",
            "Size_UL",
            "Livetime",
            "Energy_Threshold",
            "Flux_Map",
            "Flux_Map_Err",
        ]
        colnames2 = ["Source_Name", "Distance", "Distance_Min", "Distance_Max"]
        file_name = "hgps_catalog_v1.fits.gz"
        file_path = self._get_path_to_file(file_name)
        # Silencing multiple, harmless 'MergeConflictWarning's when
        # reading the fits tables
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.source_table = join(
                Table.read(file_path, hdu=1)[colnames1],
                Table.read(file_path, hdu=4)[colnames2],
                join_type="left",
                keys="Source_Name",
            )
        # Add alias
        self.source_table["flux"] = self.source_table["Flux_Map"]
        self.source_table["flux_err"] = np.sqrt(
            self.source_table["Flux_Map_Err"] ** 2
            + (0.3 * self.source_table["Flux_Map"]) ** 2
        )
        self.source_table["extent"] = self.mask_to_value(
            self.source_table["Size"], np.nan
        )
        self.source_table["extent_err"] = self.mask_to_value(
            self.source_table["Size_Err"], 1e-16
        )
        self.source_table["distance"] = self.mask_to_value(
            self.source_table["Distance"], np.nan
        )
        self.source_table["distance_min"] = self.mask_to_value(
            self.source_table["Distance_Min"], np.nan
        )
        self.source_table["distance_max"] = self.mask_to_value(
            self.source_table["Distance_Max"], np.nan
        )
        self.source_table["coordinate"] = SkyCoord(
            l=self.source_table["GLON"].quantity,
            b=self.source_table["GLAT"].quantity,
            distance=self.source_table["distance"].quantity,
            frame="galactic",
        )
        self.source_table = Table(self.source_table)
        file_name = f"hgps_map_sensitivity_{_bin_size_in_deg}deg_v1.fits.gz"
        self.sensitivity_map = Map.read(self._get_path_to_file(file_name))
        self.sensitivity_map.data = np.where(
            np.greater(self.sensitivity_map.data, 0.0),
            self.sensitivity_map.data,
            np.nan,
        )
        # Tuning parameters: stored after map loading so defaults can
        # reference self.psf which is set above.
        self.sensitivity_scale = float(sensitivity_scale)
        self.detection_psf = detection_psf if detection_psf is not None else self.psf
        logger.info("Created an instance of HGPS")

    @staticmethod
    def mask_to_value(masked_data: MaskedColumn, value: float) -> Quantity:
        return (
            np.where(~masked_data.mask, masked_data.data.data, value) * masked_data.unit
        )

    def _get_path_to_file(self, file_name: str) -> Path:
        """Return the path to the file in the default directory.

        Check if the file is already avialable and download it if it is not.

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
        # 'not greater' handles NaN correctly (unlike <=)
        outside_analysis_region = ~np.greater(map_values, 0.0)
        outside_analysis_region |= self.galactic_center_cutout_region.contains(
            sky_coord, self.sensitivity_map.geom.wcs
        )
        map_values[outside_analysis_region] = np.inf
        # sensitivity_scale < 1 lowers the bar (more detections);
        # scale > 1 raises it. Default 1.0 reproduces published HGPS.
        return (
            map_values * self.sensitivity_map.unit * self.sensitivity_scale
        ).to(INTEGRAL_PHOTON_FLUX_UNIT)

    def get_detection_threshold(
        self, sky_coord: SkyCoord, extent: u.Quantity
    ) -> u.Quantity:
        """Yields the minimum flux required for the detection of a possibly
        extended source at the given coordinates.

        Parameters
        ----------
        sky_coord : SkyCoord
            The position of the source in celestial coordinates.
        extent : Quantity
            The angular radius of the source (i.e. 1 sigma of a
            symmetric 2D-Gaussian).

        Returns
        -------
        Quantity
            The minimum integral flux >1 TeV.
        """

        return self.get_detection_threshold_for_point_sources(sky_coord) * np.sqrt(
            1 + (np.where(np.isnan(extent), 0.0 * extent.unit, extent) / self.detection_psf) ** 2
        )

    def get_detection_breakdown(
        self,
        sky_coord: SkyCoord,
        flux: u.Quantity,
        extent: u.Quantity,
    ) -> dict:
        """Break down per-source detection failure modes for diagnostics.

        Mutually exclusive categories (in priority order):
        - ``not_visible``    : source is outside the HGPS footprint or in the
                               GC cutout region.
        - ``too_extended``   : source is visible but ``extent > max_extent``.
        - ``below_pt_thresh``: visible, extent OK, but flux is below the
                               *point-source* threshold at that position.
        - ``only_ext_penalty``: would pass as a point source but the extended-
                                source correction raises the bar above the flux.
        - ``detected``       : passes all cuts.

        Parameters
        ----------
        sky_coord : SkyCoord
            Positions of the sources.
        flux : Quantity
            Integral flux >1 TeV for each source.
        extent : Quantity
            Angular radius (1-sigma Gaussian) of each source.

        Returns
        -------
        dict
            Keys are the category strings above; values are boolean arrays of
            length ``len(sky_coord)``.
        """

        ext_clean = np.where(np.isnan(extent), 0.0 * extent.unit, extent)
        visible = self.is_visible(sky_coord)
        pt_thresh = self.get_detection_threshold_for_point_sources(sky_coord)
        full_thresh = self.get_detection_threshold(sky_coord, extent)

        not_visible = ~visible
        too_extended = visible & (ext_clean > self.max_extent)
        in_footprint_ok = visible & (ext_clean <= self.max_extent)
        below_pt_thresh = in_footprint_ok & (flux < pt_thresh)
        only_ext_penalty = in_footprint_ok & (flux >= pt_thresh) & (flux < full_thresh)
        detected = in_footprint_ok & np.greater_equal(flux, full_thresh)

        return {
            "not_visible": not_visible,
            "too_extended": too_extended,
            "below_pt_thresh": below_pt_thresh,
            "only_ext_penalty": only_ext_penalty,
            "detected": detected,
        }

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

        return ~np.isinf(self.get_detection_threshold_for_point_sources(sky_coord))

    def is_detectable(
        self, sky_coord: SkyCoord, flux: u.Quantity, extent: u.Quantity
    ) -> np.ndarray:
        """Determines the detectability of a source within the HGPS.

        Parameters
        ----------
        sky_coord : SkyCoord
            The position of the source in celestial coordinates.
        flux : Quantity
            The integral flux >1 TeV.
        extent : Quantity
            The angular radius of the source (i.e. 1 sigma of a
            symmetric 2D-Gaussian).

        Returns
        -------
        bool
            The detectability.
        """

        is_detectable = np.greater_equal(
            flux, self.get_detection_threshold(sky_coord, extent)
        )
        is_detectable &= np.less_equal(
            np.where(np.isnan(extent), 0.0 * extent.unit, extent), self.max_extent
        )
        return is_detectable


class HGPS2(Survey):
    """A class to represent the second H.E.S.S. Galactic plane survey.

    This is a preliminary version based on an unpublished catalogue.
    Details of the HGPS2 are described in:
        https://hess-confluence.desy.de/confluence/pages/viewpage.action?spaceKey=HESS&title=2HGPS+preliminary+catalog .
    The data being used was downloaded from:
        - Catalogue: https://hess-confluence.desy.de/confluence/download/attachments/332300312/SeedList_merged_complex_step3.fits?version=1&modificationDate=1672924289873&api=v2
        - Sensitivity maps: https://1drv.ms/f/s!AiTtm00zHBzSiM99VHF42jtbdkrIbw?e=FXdP5W

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
        Table of HGPS sources with their properties.
    sensitivity_maps : Dict[float, Map]
        Sensitivity wrt background + fitted Fermi diffuse emisssion for different correlation radii.
    """

    def __init__(self) -> None:
        logger.info("Creating an instance of the 2HGPS")
        self.psf = 0.08 * u.deg  # average value across the field of view
        self.min_extent = (
            0.03 * u.deg
        )  # systematic bias on the size of point-like sources
        self.max_extent = (
            1.5 * u.deg
        )  # size of the largest source in the catalogue + error margin
        self.source_table = join(
            Table.read(
                Path(DEFAULT_RESOURCE_PATH, "SeedList_merged_complex_step3.fits")
            ),
            Table.read(Path(DEFAULT_RESOURCE_PATH, "hgps_v2_distances.fits")),
            join_type="left",
            keys="Name",
        )
        self.source_table = join(
            self.source_table,
            Table.read(Path(DEFAULT_RESOURCE_PATH, "hgps_v2_model_flux.fits")),
            join_type="left",
            keys="Name",
        )
        self.source_table["extent"] = self.source_table["Radius"].quantity * u.deg
        self.source_table["extent_err"] = (
            np.nan_to_num(self.source_table["err_Radius"].quantity, nan=1e-16) * u.deg
        )
        self.source_table["coordinate"] = SkyCoord(
            l=self.source_table["GLON"].quantity * u.deg,
            b=self.source_table["GLAT"].quantity * u.deg,
            distance=self.source_table["distance"].quantity,
            frame="galactic",
        )
        self.source_table["amplitude"] = (
            self.source_table["amplitude"].quantity * DIFFERENTIAL_PHOTON_FLUX_UNIT
        )
        self.source_table["err_amplitude"] = (
            self.source_table["err_amplitude"].quantity * DIFFERENTIAL_PHOTON_FLUX_UNIT
        )
        self.source_table["F_500GeV-100TeV"] = (
            self.source_table["F_500GeV-100TeV"].quantity * INTEGRAL_PHOTON_FLUX_UNIT
        )
        self.source_table["reference_energy"] = (
            self.source_table["Eref"].quantity * u.TeV
        )
        self.source_table = Table(self.source_table)
        self.reduced_table = QTable(
            {
                "coordinate": SkyCoord(
                    l=self.source_table["GLON"].quantity * u.deg,
                    b=self.source_table["GLAT"].quantity * u.deg,
                    distance=self.source_table["distance"].quantity,
                    frame="galactic",
                ),
                "amplitude": self.source_table["amplitude"].quantity,
                "reference_energy": self.source_table["Eref"].quantity * u.TeV,
                "alpha": self.source_table["alpha"],
                "beta": self.source_table["beta"],
                "extent": np.nan_to_num(self.source_table["Radius"].quantity, nan=1e-16)
                * u.deg,
            }
        )
        self.sensitivity_maps = {
            corr_radius: self._load_sensitivity_map(corr_radius)
            for corr_radius in ["0.1", "0.2", "0.4"]
        }
        logger.info("Created an instance of 2HGPS")

    @staticmethod
    def _load_sensitivity_map(corr_radius: str) -> Map:
        s_map = Map.read(
            Path(
                DEFAULT_RESOURCE_PATH,
                f"sensitivity_4sigma_{corr_radius}deg_GPS-hess12_1u_500GeV-100TeV_1_iembkg_step3.fits",
            )
        ).sum_over_axes(keepdims=False)
        s_map.quantity = np.where(
            s_map.quantity > 0.0 * s_map.unit, s_map.quantity, np.inf * s_map.unit
        )
        return s_map

    def _get_fill_factor(self, extent: u.Quantity, corr_radius: float) -> float:
        """Calculate fraction of the psf-convolved flux contained within
        the correlation radius. We assume radial symmetry.

        Parameters:
        -----------
        extent : u.Quantity
            Extent of the source.
        corr_radius : float
            Correlation radius in degree.

        Returns:
        --------
        float
            Fraction of the flux contained within the correlation radius.
        """

        # PSF-convolved extent of the source assuming a Gaussian as
        # spatial model
        extent_convolved = np.linalg.norm(
            [
                np.nan_to_num(extent).to_value("deg"),
                np.full_like(extent.to_value("deg"), self.psf.to_value("deg")),
            ],
            axis=0,
        )
        # Fraction is determined the CDF of a half normal distribution
        return halfnorm.cdf(
            np.full_like(extent_convolved, corr_radius),
            loc=np.zeros_like(extent_convolved),
            scale=extent_convolved,
        )

    def get_detection_threshold(
        self, sky_coord: SkyCoord, extent: u.Quantity
    ) -> u.Quantity:
        """Yields the minimum flux required for the detection of a possibly
        extended source at the given coordinates.

        Parameters
        ----------
        sky_coord : SkyCoord
            The position of the source in celestial coordinates.
        extent : Quantity
            The angular radius of the source (i.e. 1 sigma of a
            symmetric 2D-Gaussian).

        Returns
        -------
        Quantity
            The minimum integral flux >1 TeV.
        """

        return (
            np.min(
                [
                    s_map.get_by_coord(sky_coord, np.inf)
                    / self._get_fill_factor(np.nan_to_num(extent), float(corr_radius))
                    for corr_radius, s_map in self.sensitivity_maps.items()
                ],
                axis=0,
            )
            * INTEGRAL_PHOTON_FLUX_UNIT
        )

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

        return ~np.isinf(self.sensitivity_maps["0.1"].get_by_coord(sky_coord, np.inf))

    def is_detectable(
        self, sky_coord: SkyCoord, flux: u.Quantity, extent: u.Quantity
    ) -> np.ndarray:
        """Determines the detectability of a source within the HGPS.

        Parameters
        ----------
        sky_coord : SkyCoord
            The position of the source in celestial coordinates.
        flux : Quantity
            The integral flux >1 TeV.
        extent : Quantity
            The angular radius of the source (i.e. 1 sigma of a
            symmetric 2D-Gaussian).

        Returns
        -------
        bool
            The detectability.
        """

        is_detectable = np.greater_equal(
            flux, self.get_detection_threshold(sky_coord, extent)
        )
        is_detectable &= np.less_equal(np.nan_to_num(extent), self.max_extent)
        return is_detectable