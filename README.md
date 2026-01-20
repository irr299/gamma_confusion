# Gamma-Ray Flux Map Simulation & Peak Detection

## Overview

This analysis workflow simulates a Galactic gamma-ray flux map and demonstrates peak detection using Gammapy tools.

## What's Inside

**`gamma_map_simulation.ipynb`** - Complete workflow notebook containing:

1. **Map Setup**: Creates a 20° × 10° WCS map in Galactic coordinates centered on the Galactic Center
2. **Source Simulation**: Adds 6 Gaussian sources with varying intensities plus diffuse background
3. **Angular Resolution**: Applies 0.1° FWHM Gaussian PSF convolution to simulate telescope resolution
4. **Peak Detection**: Uses `gammapy.estimators.utils.find_peaks_in_flux_map` to identify sources
5. **Visualization**: Plots before/after PSF maps and overlays detected peaks

## Key Parameters

- **Map geometry**: 200 × 100 pixels (0.1° pixel size)
- **PSF FWHM**: 0.1° (σ ≈ 0.042°)
- **Detection threshold**: 3σ
- **Minimum peak separation**: 0.3°

## Requirements

```bash
pip install gammapy astropy numpy matplotlib
```

## Usage

Simply open and run `gamma_map_simulation.ipynb` in Jupyter. All cells are self-contained and execute sequentially.

## Output

- Interactive visualizations of flux maps (before/after PSF)
- Table of detected sources with positions and fluxes
- Optional FITS export (commented by default)

## Reference

Based on techniques from:
- Gammapy documentation: https://docs.gammapy.org/2.0/
- gammapop_student tutorial patterns
