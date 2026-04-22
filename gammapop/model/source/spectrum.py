import numpy as np
from astropy.table import QTable
from astropy.units import Quantity, Unit
from gammapy.modeling.models import LogParabolaSpectralModel
from scipy.stats import norm

from ...survey import HGPS2
from ..core import Model


class SpectrumModel(Model):
    def __init__(
        self, table: QTable, energy_min: Quantity, energy_max: Quantity
    ) -> None:
        super().__init__()
        self.table = table
        self.table["alpha_err"] = np.where(
            self.table["alpha_err"] > 0.0, self.table["alpha_err"], 1e-16
        )
        self.table["beta_err"] = np.where(
            self.table["beta_err"] > 0.0, self.table["beta_err"], 1e-16
        )
        self.emin = energy_min
        self.emax = energy_max

    def get_sample(
        self, integral_flux: Quantity, rng=np.random.default_rng()
    ) -> QTable:
        size = len(integral_flux)
        select = rng.uniform(low=0, high=len(self.table), size=size).astype(np.int32)
        alpha = np.round(
            np.clip(
                norm.rvs(
                    loc=self.table["alpha"][select],
                    scale=self.table["alpha_err"][select],
                ),
                0.0,
                None,
            ),
            4,
        ).astype(np.float32)
        beta = np.round(
            np.clip(
                norm.rvs(
                    loc=self.table["beta"][select],
                    scale=self.table["beta_err"][select],
                ),
                0.0,
                None,
            ),
            4,
        ).astype(np.float32)
        energy_reference = self.table["reference_energy"][select]
        phi_ref = 1e-12 * Unit("cm-2 s-1 TeV-1")
        phi = []
        for int_flux, alp, bet, e_ref in zip(
            integral_flux, alpha, beta, energy_reference
        ):
            amplitude = (
                phi_ref
                * int_flux
                / LogParabolaSpectralModel(
                    amplitude=phi_ref, reference=e_ref, alpha=alp, beta=bet
                ).integral(self.emin, self.emax)
            ).astype(np.float32)
            phi.append(amplitude)
        return QTable(
            {
                "amplitude": phi,
                "reference_energy": energy_reference,
                "alpha": alpha,
                "beta": beta,
            }
        )

    @classmethod
    def from_hgps(cls):
        table = QTable(
            {
                "alpha": np.array([2.3]),
                "alpha_err": np.array([0.0]),
                "beta": np.array([0.0]),
                "beta_err": np.array([0.0]),
                "reference_energy": np.array([1.0]) * Unit("TeV"),
            }
        )
        return cls(table, 1.0 * Unit("TeV"), 10.0 * Unit("TeV"))

    @classmethod
    def from_hgps2(cls):
        survey = HGPS2()
        # filter odd sources from the sample
        select = (survey.source_table["alpha"] > 0.2) & (
            survey.source_table["alpha"] < 4.8
        )
        t_ = survey.source_table[select]
        # spectrum table
        table = QTable(
            {
                "alpha": t_["alpha"],
                "alpha_err": t_["err_alpha"],
                "beta": t_["beta"],
                "beta_err": t_["err_beta"],
                "reference_energy": t_["Eref"] * Unit("TeV"),
            }
        )
        return cls(table, 500 * Unit("GeV"), 100 * Unit("TeV"))
