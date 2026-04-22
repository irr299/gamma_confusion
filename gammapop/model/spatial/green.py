from .core import SymmetricDiscSpatialModel


class GreenSpatialModel(SymmetricDiscSpatialModel):
    """Symmetric disc model from Green 2015"""

    def __init__(self, rmaxInKpc=30, resolutionInKpc=0.01, fraction_run_away=0.0):
        super().__init__(
            r_offInKpc=0.0,
            R_sunInKpc=8.5,
            alpha=1.09,
            beta=3.87,
            z_scaleInKpc=0.083,
            rmaxInKpc=rmaxInKpc,
            resolutionInKpc=resolutionInKpc,
            fraction_run_away=fraction_run_away,
        )
