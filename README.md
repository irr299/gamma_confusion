# Gamma Confusion Analysis

Analysis of gamma-ray source confusion using population synthesis with the HGPS survey.

## Quick Start

### 1. Clone the repository

```bash
git clone git@github.com:irr299/gamma_confusion.git
cd gamma_confusion
```

### 2. Create virtual environment

**Using venv (recommended):**
```bash
python3 -m venv env
source env/bin/activate      # On macOS/Linux
# env\Scripts\activate       # On Windows
```

**Or using conda:**
```bash
conda create -n gamma_env python=3.11
conda activate gamma_env
```

### 3. Install dependencies

```bash
pip install -e .
```

This installs the `gammapop` package and all dependencies (gammapy, astropy, numpy, etc).

### 4. Run the notebook

```bash
jupyter notebook gamma_analysis.ipynb
```

Or open in VS Code / Cursor with the Jupyter extension.

## Project Structure

```
gamma_confusion/
├── gamma_analysis.ipynb   # Main analysis notebook
├── gammapop/              # Population synthesis package
│   ├── model/             # Source population models
│   ├── survey/            # Survey definitions (HGPS, Fermi)
│   │   └── hess.py        # Modified HGPS survey class
│   ├── utils/             # Utilities (skymap, visualization)
│   └── optimize/          # Fitting routines
├── resources/             # FITS data files
│   ├── hgps_catalog_v1.fits.gz
│   ├── hgps_map_sensitivity_*.fits.gz
│   └── ...
└── pyproject.toml         # Dependencies
```

## Dependencies

- Python >= 3.10
- gammapy >= 1.0.1
- astropy >= 5.2.2
- numpy, scipy, matplotlib
- numba, tables, tqdm

## Notes

- The `resources/` folder contains ~100MB of FITS data files required for the analysis
- The `gammapop/survey/hess.py` contains custom modifications for source confusion analysis
