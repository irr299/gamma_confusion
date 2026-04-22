# Code Changes — April 2026
**Topic:** Understanding the low detection efficiency (~7.5%) of simulated gamma-ray sources in the H.E.S.S. Galactic Plane Survey (HGPS) 

---

## 1. `hess.py` — HGPS Class Changes
 
### 1a. New parameter: `sensitivity_scale`

```python
HGPS(sensitivity_scale=1.0)   # default — matches published HGPS
HGPS(sensitivity_scale=0.5)   # halves the threshold → more detections
HGPS(sensitivity_scale=2.0)   # doubles the threshold → fewer detections
```

It multiplies the raw sensitivity map pixel value before the unit conversion to integral flux. This is a direct, control parameter to explore how sensitive the detection count is to the absolute calibration of the HGPS threshold.

The simulated source luminosities are drawn from a power-law model and converted to flux via `F = L / (4π d²)`, with no energy band correction. There is therefore an implicit systematic offset between the simulation's flux values and the HGPS "integral flux >1 TeV" sensitivity. `sensitivity_scale` lets you quantify and correct for this offset.

The graph second new graph gives better understanding actually.

#### Restored method: `get_detection_threshold_for_point_sources` (Initially removed and later restored)

This reads the HGPS sensitivity map at a given sky position and returns the minimum detectable integral flux (>1 TeV) for a point source at that location. It has been restored with the `sensitivity_scale` modification described below.


### 1b. New parameter: `detection_psf`

```
HGPS(detection_psf=0.08 * u.deg)   # default — average HGPS PSF
HGPS(detection_psf=0.15 * u.deg)   # softer penalty for extended sources
```

It controls the PSF value used only in the extended-source threshold correction:

```
threshold_extended = threshold_point_source × sqrt(1 + (extent / detection_psf)²)
```

It is the PSF value plugged into the formula above. Nothing more, nothing less.

In the original code it was hardcoded as self.psf = 0.08°. The problem with that is:

self.psf = 0.08° is an average across the whole HGPS footprint. The actual PSF varies — it is sharper near the telescope pointing centres and broader at the edges. Using one fixed value applies the same penalty everywhere.

The psf attribute is also used for other things (e.g., in the notebook's min_distance calculation and extent comparisons). Separating detection_psf from psf lets you tune the threshold formula independently without side effects.

The fixed value `psf = 0.08°` used in the original code is an average across the whole survey footprint. For a source with `extent = 0.2°`, this multiplier is **2.7×** — the source must be nearly 3 times brighter than a point source at the same position to be detected. By decoupling `detection_psf` from `self.psf`, we can study how sensitive the detection count is to the assumed PSF without changing any other calculation.


**Extent penalty at different PSF assumptions:**

| Source extent | psf = 0.08° (default) | psf = 0.12° | psf = 0.15° |
|:---:|:---:|:---:|:---:|
| 0.0° | 1.0× | 1.0× | 1.0× |
| 0.1° | 1.6× | 1.3× | 1.2× |
| 0.2° | 2.7× | 1.9× | 1.6× |
| 0.4° | 5.1× | 3.5× | 2.8× |


**Vary `detection_psf`** (`0.08`, `0.12`, `0.15` deg) and re-run the comparison to quantify how much of the detection inefficiency is an artefact of the simplified PSF-averaged threshold formula vs. a genuine flux deficit.

### 1c. New method: `get_detection_breakdown(sky_coord, flux, extent)`

This gives the understanding of the source non-detection.

Returns a Python `dict` of mutually exclusive boolean arrays, one per source, classifying each non-detected source by the exact reason it was missed:

| Category | Meaning |
|----------|---------|
| `not_visible` | Outside HGPS footprint **or** inside the GC cutout rectangle (l ∈ [−1°, 1.5°], b ∈ [−0.5°, 0.5°]) |
| `too_extended` | Inside footprint but angular extent > `max_extent` (1.0°) |
| `below_pt_thresh` | Extent is fine but flux < point-source sensitivity at that sky position |
| `only_ext_penalty` | Would be detected as a point source, but the extended-source correction raises the bar above the source flux |
| `detected` | Passes all cuts |

These categories are strictly mutually exclusive and collectively exhaustive (all 1000 sources fall into exactly one). Usage:

```python
breakdown = survey.get_detection_breakdown(
    population["coordinate"],
    population["flux"],
    population["extent"],
)
for label, mask in breakdown.items():
    print(f"{label:20s}: {mask.sum()} ({100*mask.mean():.1f}%)")
```

---

## 2. `gamma_peak_finder.ipynb` — Notebook Changes

### 2a. Detection breakdown (diagnostic) - (Cell 18)

Calls `survey.get_detection_breakdown()` using the **true** population extents from `PopulationModel`. 

This directly answers: *is the detection loss dominated by sources being outside the footprint, by faint fluxes, or by the extended-source PSF penalty?*

The graph also gives better overview.

### 2b. Sensitivity scale sweep - (Cell 19) !!! 
This is bit cool thing to get understanding. 

Loops over **37 values** of `sensitivity_scale` from 0.2 to 2.0. For each value, creates a fresh `HGPS(sensitivity_scale=s)` instance and counts how many of the 1000 simulated sources would be detected. Plots the curve and marks:

- A horizontal dashed line at **78** — the number of sources in the published HGPS catalogue. (Based on the paper)
- A vertical marker at the `sensitivity_scale` value where the simulated count crosses 78.

This is the main calibration tool: it tells us how far *the simulation's effective flux scale* is from what the real HGPS sensitivity map expects.

Again this is to find the `sensitivity_scale` that would reproduce 78 real HGPS detections. If the crossing happens at `scale ≈ 0.3–0.5`, the simulation is producing sources that are systematically fainter than what the real Galactic population needs to be — pointing to the luminosity function parameters. If the crossing is near `1.0`, the model is well-calibrated.