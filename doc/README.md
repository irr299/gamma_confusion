# Gamma-Ray Flux Map Simulation & Peak Detection

## Overview

This analysis workflow simulates a Galactic gamma-ray flux map and demonstrates peak detection using Gammapy tools.


## Summary

**Workflow implemented (following gammapop tutorials):**
1. Created a population model with spatial distribution (Reid spiral + Sormani bar) and source properties (power-law luminosity & radius)
2. Simulated 1000 gamma-ray sources from the model
3. Applied HGPS survey sensitivity to select detectable sources
4. Generated sky map using `get_sky_map()` utility (energy range: 1-10 TeV, resolution: 0.1°)
5. Applied 0.1° FWHM Gaussian PSF to model angular resolution
6. Detected peaks using `find_peaks_in_flux_map` (3σ threshold, 0.3° minimum separation)
7. Visualized results and created source catalog


**Key parameters:**
- Population size: 1000 sources
- Detectable sources: ~7% (typical for HGPS)
- Energy range: 1-10 TeV
- Sky region: l ∈ [-100°, 70°], b ∈ [-5°, 5°]
- Map resolution: 0.1°
- PSF FWHM: 0.1° (σ ≈ 0.042°)
- Detection threshold: 3σ
- Minimum peak separation: 0.3°


## Output

- Interactive visualizations of flux maps (before/after PSF)
- Table of detected sources with positions and fluxes
- Optional FITS export (commented by default)