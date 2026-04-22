"""This module provides an interface to gamma-ray source catalogues.

Catalogues are represented by classes, which provide access to the
sources and their measured properties as well as methods to determine
the visibility and detectability of a given source wrt the sensitivity
range of an instrument.

Author: Constantin Steppa (2023)
"""

from .fermi import FGL
from .hess import HGPS, HGPS2
