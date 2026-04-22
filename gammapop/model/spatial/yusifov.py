from .core import SymmetricDiscSpatialModel


class YusifovSpatialModel(SymmetricDiscSpatialModel):
    """Symmetric disc model from Yusifov et al. 2004"""

    def __init__(self, rmaxInKpc=30, resolutionInKpc=0.01, fraction_run_away=0.0):
        super().__init__(
            r_offInKpc=0.55,
            R_sunInKpc=8.5,
            alpha=1.64,
            beta=4.01,
            z_scaleInKpc=0.18,
            rmaxInKpc=rmaxInKpc,
            resolutionInKpc=resolutionInKpc,
            fraction_run_away=fraction_run_away,
        )
