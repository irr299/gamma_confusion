from typing import Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import Galactocentric, SkyCoord, concatenate
from astropy.table import MaskedColumn, QTable, Table, vstack
from gammapy.modeling.models import PowerLaw2SpectralModel

from ..survey import HGPS, HGPS2
from ..survey.core import Survey
from .convert import *


def insert_hgps_sources(sim_table: Table) -> Table:
    """Replace simulated sources by actual sources from the HGPS.

    For real sources without distance estimate we sample values that
    match the simulated source distribution.

    Parameters:
    -----------
    sim_table : Table
        A table of simulated sources.

    Returns:
    --------
    Table
        The table with real and simulated sources.
    """

    survey = HGPS()
    _survey_table = homogenize_coordinate_frames(survey.source_table, sim_table)
    _survey_table["Name"] = _survey_table["Source_Name"]
    _survey_table["source_class"] = _survey_table["Source_Class"]
    _survey_table["Asso_Name"] = MaskedColumn(
        data=_survey_table["Identified_Object"],
        name="Asso_Name",
        mask=_survey_table["Identified_Object"].mask,
        fill_value="",
    )
    _survey_table["amplitude"] = u.Quantity(
        [
            PowerLaw2SpectralModel.evaluate(
                energy=1.0 * u.TeV,
                index=2.3,
                amplitude=amp,
                emin=1.0 * u.TeV,
                emax=10.0 * u.TeV,
            )
            for amp in _survey_table["flux"].quantity
        ]
    )
    _survey_table["reference_energy"] = np.ones(len(_survey_table)) * u.TeV
    _survey_table["alpha"] = np.full(len(_survey_table), 2.3)
    _survey_table["beta"] = np.zeros(len(_survey_table))
    _survey_table["extent"] = np.nan_to_num(_survey_table["extent"])
    _sim_table = sim_table.copy()
    # replace simulated sources by real ones
    real_table, _sim_table = replace_sim_by_real(
        _survey_table, _sim_table, survey.min_extent
    )
    # shuffle the remaining simulated sources such that there are no more
    # detectable sources within the HGPS field of view (there shall not be
    # more than the real detected ones).
    _sim_table = switch_detectable_sim_sources(_sim_table, survey)
    # fill missing values with NaNs.
    _sim_table["name"] = np.array([f"SIM_{iii}" for iii in range(len(_sim_table))])
    _sim_table["common name"] = np.array(["" for _ in range(len(_sim_table))])
    _sim_table["type"] = np.array(["simulated" for _ in range(len(_sim_table))])
    _sim_table["source_class"] = np.array(["simulated" for _ in range(len(_sim_table))])
    _sim_table["amplitude"] = u.Quantity(
        [
            PowerLaw2SpectralModel.evaluate(
                energy=1.0 * u.TeV,
                index=2.3,
                amplitude=amp,
                emin=1.0 * u.TeV,
                emax=10.0 * u.TeV,
            )
            for amp in _sim_table["flux"]
        ]
    )
    _sim_table["reference_energy"] = np.ones(len(_sim_table)) * u.TeV
    _sim_table["alpha"] = np.full(len(_sim_table), 2.3)
    _sim_table["beta"] = np.zeros(len(_sim_table))

    return vstack([real_table, _sim_table])


def insert_hgps2_sources(sim_table: Table) -> Table:
    """Replace simulated sources by actual sources from the 2HGPS.

    For real sources without distance estimate we sample values that
    match the simulated source distribution.

    Parameters:
    -----------
    sim_table : Table
        A table of simulated sources.

    Returns:
    --------
    Table
        The table with real and simulated sources.
    """

    survey = HGPS2()
    _sim_table = sim_table.copy()
    _real_table = homogenize_coordinate_frames(survey.source_table, _sim_table)
    _real_table["source_class"] = np.full(len(_real_table), "HGPS2 source")
    # replace simulated sources by real ones
    real_table, _sim_table = replace_sim_by_real(
        _real_table, _sim_table, survey.min_extent
    )
    # shuffle the remaining simulated sources such that there are no more
    # detectable sources within the HGPS field of view (there shall not be
    # more than the real detected ones).
    _sim_table = switch_detectable_sim_sources(_sim_table, survey)
    # fill missing values with NaNs.
    _sim_table["name"] = np.array([f"SIM_{iii}" for iii in range(len(_sim_table))])
    _sim_table["common name"] = np.array(["" for _ in range(len(_sim_table))])
    _sim_table["type"] = np.array(["simulated" for _ in range(len(_sim_table))])
    _sim_table["amplitude"] = (
        np.full(len(_sim_table), np.nan) * real_table["amplitude"].unit
    )
    _sim_table["reference_energy"] = (
        np.full(len(_sim_table), np.nan) * real_table["reference_energy"].unit
    )
    _sim_table["alpha"] = np.full(len(_sim_table), np.nan)
    _sim_table["beta"] = np.full(len(_sim_table), np.nan)

    return vstack([real_table, _sim_table])


def homogenize_coordinate_frames(real_table: Table, sim_table: Table) -> Table:
    """Transform the coordinates in the table of real sources such that the
    frame is consistent with that of the simulated source, e.g. wrt z_sun.

    Parameters:
    -----------
    real_table : Table
        Table of real sources.
    sim_table : Table
        Table of simulated sources.

    Returns:
    --------
    Table
        Table of real sources with consistent coordinate frame.
    """
    new_table = real_table.copy()
    new_table["coordinate"] = transform_real_coordinates_to_sim_frame(
        new_table["coordinate"], sim_table["coordinate"]
    )
    return new_table


def transform_real_coordinates_to_sim_frame(
    real_coord: SkyCoord,
    sim_coord: SkyCoord,
) -> SkyCoord:
    """Transforming the coordinates of real sources such that the frame is
    consistent with that of the simulated source, e.g. wrt z_sun.

    Parameters:
    -----------
    real_coord : SkyCoord
        Coordinates of the real sources.
    sim_coord : SkyCoord
        Coordinates of the simulated sources.

    Returns:
    --------
    SkyCoord
        Coordinates of the real sources in the frame of the simulated sources.
    """
    coord = SkyCoord(
        l=real_coord.l,
        b=real_coord.b,
        distance=real_coord.distance,
        frame=sim_coord.frame,
    )
    for k, v in sim_coord.__dict__.items():
        if k != "_sky_coord_frame" and k != "length":
            setattr(coord, k, v)
    return coord


def replace_sim_by_real(
    real_sources: Table, sim_sources: Table, min_extent: u.Quantity
) -> Tuple[QTable, QTable]:
    """Replace simulated sources by real ones.

    Choose simulated sources that are the best match to the real one for
    replacement.

    Parameters:
    -----------
    real_sources : Table
        The table of real sources.
    sim_sources : Table
        The table of simulated sources.
    min_extent : u.Quantity
        The minimum angular extent a source in the corresponding survey
        must exhibit to be spatially resolved.

    Returns:
    --------
    Tuple[QTable, QTable]
        The table of real sources with distances and radii from either
        observation or simulation in accordance with the population model
        The table of remaining simulated sources.
    """
    sim_table = sim_sources.copy()
    sources = []
    coord = real_sources["coordinate"]
    flux = real_sources["flux"].quantity
    extent = real_sources["extent"].quantity
    for index in range(len(real_sources)):
        source, sim_table = extract_source(
            coord[index],
            flux[index],
            extent[index],
            sim_table,
            min_extent,
        )
        sources.append(source)
    real_coord = transform_real_coordinates_to_sim_frame(
        concatenate(
            [
                transform_real_coordinates_to_sim_frame(
                    s["coordinate"], sim_table["coordinate"]
                )
                for s in sources
            ]
        )
        .transform_to(get_frame_from_sim_table(sim_table))
        .galactic,
        sim_table["coordinate"],
    )
    real_table = QTable(
        {
            "name": real_sources["Name"],
            "source_class": real_sources["source_class"],
            "common name": np.where(
                real_sources["Asso_Name"].mask,
                "",
                np.array(
                    [x.decode().upper() for x in real_sources["Asso_Name"].data.data]
                ),
            ),
            "type": np.array(["real" for _ in range(len(sources))]),
            "coordinate": real_coord,
            "luminosity": np.stack([s["luminosity"] for s in sources]),
            "radius": np.stack([s["radius"] for s in sources]),
            "flux": np.stack([s["flux"] for s in sources]),
            "extent": np.stack([s["extent"] for s in sources]),
            "amplitude": real_sources["amplitude"].quantity,
            "reference_energy": real_sources["reference_energy"].quantity,
            "alpha": real_sources["alpha"],
            "beta": real_sources["beta"],
        }
    )
    return real_table, sim_table


def extract_source(
    source_coordinate: SkyCoord,
    source_flux: Quantity,
    source_extent: Quantity,
    sim_table: Table,
    min_extent: Quantity,
) -> Tuple[dict, Table]:
    """Find the closest match to a real source in the simulation.

    Extract missing information for the real source from the simulated
    sample without altering its global distribution.

    Parameters:
    -----------
    source_coordinate : SkyCoord
        The coordinate of the real source.
    source_flux : Quantity
        The flux of the real source.
    source_extent : Quantity
        The angular extent of the real source.
    sim_table : Table
        The table of simulated source.
    min_extent : Quantity
        The minimum angular extent a source in the corresponding survey
        must exhibit to be spatially resolved.

    Returns:
    --------
    Tuple[dict, Table]
        A dictionary with the properties of the real sources, including
        missing values filled by values sample from the simulation.
        The table of remaining simulated sources.
    """

    # If the source has no distance estimate, we use the distance of the
    # simulated source which is closest to the real source's line of sight.
    # Otherwise, we use the real distance of the source and check for the
    # simulated source with the smallest 3d distance.
    if np.isnan(source_coordinate.distance):
        coord, index_c = get_coordinate_from_line_of_sight_distance(
            source_coordinate,
            sim_table["coordinate"],
            get_frame_from_sim_table(sim_table),
        )
    else:
        coord, index_c = get_coordinate_from_3d_distance(
            source_coordinate, sim_table["coordinate"]
        )

    # If the source is point like, we sample a radius from the simulated
    # sources which corresponds to an angular extent at the source
    # position that is below the resolution of the survey.
    # Otherwise, use the measured extent of the source.
    extent = source_extent
    if np.isnan(extent):
        luminosity, radius, index_lr = get_properties_without_extent(
            coord, source_flux, min_extent, sim_table
        )
        extent = radius_to_extent(radius, coord.distance)
    else:
        luminosity, radius, index_lr = get_properties_with_extent(
            coord, source_flux, extent, sim_table
        )
    source = {
        "coordinate": coord,
        "luminosity": luminosity,
        "radius": radius,
        "flux": source_flux,
        "extent": extent,
    }
    _sim_table = get_source_extracted_sim_table(sim_table, index_c, index_lr)
    return source, _sim_table


def coordinate_to_vector(coordinate: SkyCoord, frame: Galactocentric) -> np.ndarray:
    """Transforms a SkyCoord into a 3D Cartesian vector with (x,y,z) at axis=1.

    Parameters:
    -----------
    coordinate : SkyCoord
        Coordinate that is transformable to the considered frame
        (i.e. has distance).
    frame : Galactocentric
        The frame to transform to, in particular the position of the Sun
        should be specified.

    Returns:
    --------
    np.ndarray
        The Cartesian vector for the coordinate.
    """

    _c = coordinate.transform_to(frame)
    return np.stack(
        [_c.x.to_value("kpc"), _c.y.to_value("kpc"), _c.z.to_value("kpc")], axis=-1
    )


def get_sun_position_from_frame(frame: Galactocentric) -> np.ndarray:
    """Get the position of the Sun in the considered frame as
    3D Cartesian vector.

    Parameters:
    -----------
    frame : Galactocentric
        The frame to use.

    Returns:
    --------
    np.ndarray
        The Cartesian vector for the position of the Sun.
    """

    # To comply with the astropy definition, the Sun is at negative x values.
    return np.array(
        [-frame.galcen_distance.to_value("kpc"), 0.0, frame.z_sun.to_value("kpc")]
    )


def get_line_of_sight_unit_vector(coord: SkyCoord, frame: Galactocentric) -> np.ndarray:
    """Get a unit vector in 3D Cartesian coordinates in direction of a source.

    Parameters:
    -----------
    coord : SkyCoord
        Coordinates of the source(s).
    frame : Galactocentric
        The frame to transform to, in particular the position of the Sun
        should be specified.

    Returns:
    --------
    np.ndarray
        Unit vector(s) in the direction of the source(s).
    """
    vector = coordinate_to_vector(
        SkyCoord(l=coord.l, b=coord.b, distance=1.0 * u.kpc, frame="galactic"), frame
    )
    vector -= get_sun_position_from_frame(frame)
    vector /= np.linalg.norm(vector, axis=-1)
    return vector


def get_coordinate_from_line_of_sight_distance(
    real_coord: SkyCoord, sim_coord: SkyCoord, frame: Galactocentric
) -> Tuple[SkyCoord, int]:
    """Sample the position of source without distance estimate such that,
    it follows the source spatial distribution underlying the simulated
    sample.

    The distance is taken as the projected distance along the line of sight
    of tje simulated source that is closest to that line of sight.

    Parameters:
    -----------
    real_coord : SkyCoord
        The sky coordinate of the real source (without distance estimate).
    sim_coord : SkyCoord
        The coordinates of the simulated sources.
    frame : Galactocentric
        The frame to transform to, in particular the position of the Sun
        should be specified..

    Returns:
    --------
    Tuple[SkyCoord, int]
        The sky coordinate of the real source with a simulated distance.
        The index of the source in the simulated sample that is closest
        to the line of sight.
    """

    los_vector = get_line_of_sight_unit_vector(real_coord, frame)
    sim_vector = coordinate_to_vector(sim_coord, frame) - get_sun_position_from_frame(
        frame
    )
    projected_distance = np.dot(sim_vector, los_vector)
    projected_position = (
        get_sun_position_from_frame(frame) + projected_distance[..., None] * los_vector
    )
    distance_to_los = np.linalg.norm(sim_vector - projected_position, axis=-1)
    select = projected_distance >= 0.0
    _index = np.argmin(distance_to_los[select])
    index = np.arange(len(sim_coord))[select][_index]
    return (
        SkyCoord(
            l=real_coord.l,
            b=real_coord.b,
            distance=projected_distance[index] * u.kpc,
            frame="galactic",
        ),
        index,
    )


def get_coordinate_from_3d_distance(
    real_coord: SkyCoord, sim_coord: SkyCoord
) -> Tuple[SkyCoord, int]:
    """Returns the actual position of the real source and the index of
    the source in the simulated sample that is closest to the real source.

    Parameters:
    -----------
    real_coord : SkyCoord
        Coordinate of the real source.
    sim_coord : SkyCoord
        Coordinates of the simulated sources.

    Returns:
    --------
    Tuple[SkyCoord, int]
        _description_.
    """

    index = np.argmin(real_coord.separation_3d(sim_coord))
    return real_coord, index


def get_frame_from_sim_table(table: Table) -> Galactocentric:
    """Get a Galactocentric frame from the meta data of the table of
    simulated sources.

    Parameters:
    -----------
    table : Table
        Table of simulated sources.

    Returns:
    --------
    Galactocentric
        The frame corresponding the one used to sample the synthetic population.
    """

    return Galactocentric(
        galcen_distance=np.abs(table.meta["SUN_X"]) * u.Unit(table.meta["SUN_UNIT"]),
        z_sun=table.meta["SUN_Z"] * u.Unit(table.meta["SUN_UNIT"]),
    )


def get_property_vector(luminosity: Quantity, radius: Quantity) -> np.ndarray:
    """Get a combined 2D vector of luminosity and radius both in log10 space.

    Parameters:
    -----------
    luminosity : Quantity
        Luminosity of the source(s).
    radius : Quantity
        Radius of the source(s).

    Returns:
    --------
    np.ndarray
        The 2D vectors of the combined quantities.
    """

    return np.stack(
        [np.log10(luminosity.to_value("s-1")), np.log10(radius.to_value("pc"))], axis=-1
    )


def get_properties_with_extent(
    real_coord: SkyCoord, real_flux: Quantity, real_extent: Quantity, sim_table: Table
) -> Tuple[Quantity, Quantity, int]:
    """Returns the luminosity and radius of the real source and the index of
    the source in the simulated sample that yields the closest match to the
    real values.

    Parameters:
    -----------
    real_coord : SkyCoord
        Coordinate of the real source.
    real_flux : Quantity
        Integral flux of the real source.
    real_extent : Quantity
        Angular extent of the real source.
    sim_table : Table
        Table of simulated sources.

    Returns:
    --------
    Tuple[Quantity, Quantity, int]
        The luminosity of the real source.
        The radius of the real source.
        The index of the source in the simulated sample that matches the
        real source values best.
    """

    real_luminosity = flux_to_luminosity(real_flux, real_coord.distance)
    real_radius = extent_to_radius(real_extent, real_coord.distance)
    real_properties = get_property_vector(real_luminosity, real_radius)
    sim_properties = get_property_vector(sim_table["luminosity"], sim_table["radius"])
    index = np.argmin(np.linalg.norm(sim_properties - real_properties), axis=-1)
    return real_luminosity, real_radius, index


def get_properties_without_extent(
    real_coord: SkyCoord, real_flux: Quantity, min_extent: Quantity, sim_table: Table
) -> Tuple[Quantity, Quantity, int]:
    """Returns the luminosity of a point-like source and a simulated radius that
    complies with angular resolution of the survey. Also returns the index of
    the simulated sources that matches the values of the real source best.

    Parameters:
    -----------
    real_coord : SkyCoord
        Coordinate of the real source.
    real_flux : Quantity
        Integral flux of the real source.
    min_extent : Quantity
        Angular resolution of the survey.
    sim_table : Table
        Table of simulated sources.

    Returns:
    --------
    Tuple[Quantity, Quantity, int]
        The luminosity of the real source.
        The radius of the real source.
        The index of the source in the simulated sample that matches the
        real source values best..
    """

    # derive an upper limit on the real radius by assuming a maximum
    # angular extent corresponding to the resolution of the survey
    radius_ul = extent_to_radius(min_extent, real_coord.distance)
    select = sim_table["radius"] < radius_ul
    # if there are simulated sources in the sample with smaller radii
    # than the upper limit, select the one with the smallest difference
    # in luminosity to the real source and take its radius as the one of
    # the real source
    if select.sum():
        real_luminosity = flux_to_luminosity(real_flux, real_coord.distance)
        _index = np.argmin(
            np.abs(
                sim_table[select]["luminosity"].to_value("s-1")
                - real_luminosity.to_value("s-1")
            )
        )
        real_radius = sim_table[select][_index]["radius"]
        index = np.arange(len(sim_table))[select][_index]
    # If there is no simulated source with a smaller radius, assume the
    # angular resolution of the survey as actual source extent and
    # proceed as in `get_properties_with_extent`
    else:
        real_luminosity, real_radius, index = get_properties_with_extent(
            real_coord, real_flux, min_extent, sim_table
        )
    return real_luminosity, real_radius, index


def get_source_extracted_sim_table(
    sim_table: Table, coordinate_index: int, property_index: int
) -> Table:
    """Return the table of simulated sources removed by one source,
    that matches a real source.

    There are two indices, one for the source that matches the real
    source best wrt the position and one index for the source that
    matches the real source best wrt to source luminosity/radius. We
    will switch the values of luminosity/radius between the two simulated
    sources and then remove only one source.

    Parameters:
    -----------
    sim_table : Table
        Table of simulated sources.
    coordinate_index : int
        Index of the source that yields the best match wrt source position.
    property_index : int
        Index of the source that yields the best match wrt luminosity/radius.

    Returns:
    --------
    Table
        Reduced table.
    """

    _sim_table = sim_table.copy()
    _sim_table[property_index]["luminosity"] = _sim_table[coordinate_index][
        "luminosity"
    ]
    _sim_table[property_index]["radius"] = _sim_table[coordinate_index]["radius"]
    _sim_table[property_index]["flux"] = luminosity_to_flux(
        _sim_table[property_index]["luminosity"],
        _sim_table[property_index]["coordinate"].distance,
    )
    _sim_table[property_index]["extent"] = radius_to_extent(
        _sim_table[property_index]["radius"],
        _sim_table[property_index]["coordinate"].distance,
    )
    select = np.arange(len(_sim_table)) != coordinate_index
    return _sim_table[select]


def switch_detectable_sim_sources(sim_table: Table, survey: Survey) -> Table:
    """Switch luminosities and radius among sources in the sample of simulated
    sources until no source would be detectable in the survey.

    Once real source are included in the simulated population, among the
    remaining simulated sources there should be no detectable source since
    this would create an excess of detectable sources that contradicts
    the survey result and the model.

    Parameters:
    -----------
    sim_table : Table
        Table of simulated sources.
    survey : Survey
        Survey the sample of real sources is based on.

    Returns:
    --------
    Table
        Resampled table of simulated sources.
    """

    # first, select only sources that lie within the field of view of
    # the survey. If we don't do this, we might push all detectable
    # sources outside the field of view of the survey and, thus, create
    # an excess elsewhere.
    is_visible = survey.is_visible(sim_table["coordinate"])
    _sim_table = sim_table[is_visible].copy()
    # Now, iterate of the detectable source and swap their luminosity/radius
    # with another sources such that both are not detectable.
    is_detectable = survey.is_detectable(
        _sim_table["coordinate"], _sim_table["flux"], _sim_table["extent"]
    )
    for _ in range(is_detectable.sum()):
        idx = np.arange(len(is_detectable))[is_detectable][0]
        can_be_switched = ~survey.is_detectable(
            _sim_table["coordinate"][idx],
            luminosity_to_flux(
                _sim_table["luminosity"], _sim_table["coordinate"].distance[idx]
            ),
            radius_to_extent(
                _sim_table["radius"], _sim_table["coordinate"].distance[idx]
            ),
        ) & ~survey.is_detectable(
            _sim_table["coordinate"],
            luminosity_to_flux(
                _sim_table["luminosity"][idx], _sim_table["coordinate"].distance
            ),
            radius_to_extent(
                _sim_table["radius"][idx], _sim_table["coordinate"].distance
            ),
        )
        assert (
            can_be_switched.sum() > 0
        ), "Couldn't find a switch partner that would be undetected."
        idx_switch = np.random.choice(
            np.arange(len(can_be_switched))[can_be_switched], size=1
        )[0]
        coord_tmp = _sim_table["coordinate"][idx]
        _sim_table["coordinate"][idx] = _sim_table["coordinate"][idx_switch]
        _sim_table["coordinate"][idx_switch] = coord_tmp
        is_detectable = survey.is_detectable(
            _sim_table["coordinate"],
            luminosity_to_flux(
                _sim_table["luminosity"], _sim_table["coordinate"].distance
            ),
            radius_to_extent(_sim_table["radius"], _sim_table["coordinate"].distance),
        )
        if is_detectable.sum() == 0:
            break
    _sim_table["flux"] = luminosity_to_flux(
        _sim_table["luminosity"], _sim_table["coordinate"].distance
    )
    _sim_table["extent"] = radius_to_extent(
        _sim_table["radius"], _sim_table["coordinate"].distance
    )
    return vstack([_sim_table, sim_table[~is_visible]])
