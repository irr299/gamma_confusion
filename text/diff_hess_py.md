# Code Changes — `hess.py` vs `hess_og.py` (Original)

**Summary:** Modifications to the HGPS class to add tunability and diagnostics for understanding low detection efficiency in simulated gamma-ray source populations.

The HGPS2 class is unchanged.

---

## 1. New Parameter: `sensitivity_scale`

```python
HGPS(sensitivity_scale=1.0)   # default — matches published HGPS
HGPS(sensitivity_scale=0.5)   # halves the threshold → more detections
HGPS(sensitivity_scale=2.0)   # doubles the threshold → fewer detections
```

**What it does:** Multiplies the raw sensitivity map value before converting to integral flux.

**Original code:**
```python
return (map_values * self.sensitivity_map.unit).to(INTEGRAL_PHOTON_FLUX_UNIT)
```

**Modified code:**
```python
return (map_values * self.sensitivity_map.unit * self.sensitivity_scale).to(INTEGRAL_PHOTON_FLUX_UNIT)
```

**Why it matters:** The simulated source fluxes (`F = L / 4πd²`) have an implicit systematic offset from the HGPS "integral flux >1 TeV" definition. This parameter lets you quantify and correct for that offset without editing code.

---

## 2. New Parameter: `detection_psf`

```python
HGPS(detection_psf=0.08 * u.deg)   # default — average HGPS PSF
HGPS(detection_psf=0.15 * u.deg)   # softer penalty for extended sources
```

**What it does:** Controls the PSF value used only in the extended-source threshold correction formula:

```
threshold_extended = threshold_point × sqrt(1 + (extent / detection_psf)²)
```

**Original code:** Used `self.psf` (hardcoded 0.08°) for everything.

**Modified code:** Uses separate `self.detection_psf` for the threshold formula only.

**Why it matters:** 
- `psf = 0.08°` is an average across the whole HGPS footprint — the actual PSF varies spatially.
- The `psf` attribute is also used elsewhere (e.g., `min_distance` calculations). Separating `detection_psf` avoids side effects when tuning.

**Extent penalty at different PSF assumptions:**

| Source extent | psf = 0.08° (default) | psf = 0.12° | psf = 0.15° |
|:---:|:---:|:---:|:---:|
| 0.0° | 1.0× | 1.0× | 1.0× |
| 0.1° | 1.6× | 1.3× | 1.2× |
| 0.2° | 2.7× | 1.9× | 1.6× |
| 0.4° | 5.1× | 3.5× | 2.8× |

A source with `extent = 0.2°` using the default PSF must be **2.7× brighter** than a point source to be detected.

---

## 3. New Method: `get_detection_breakdown(sky_coord, flux, extent)`

Returns a `dict` of mutually exclusive boolean arrays classifying each source by the exact reason it was missed:

| Category | Meaning |
|----------|---------|
| `not_visible` | Outside HGPS footprint **or** inside GC cutout (l ∈ [−1°, 1.5°], b ∈ [−0.5°, 0.5°]) |
| `too_extended` | Inside footprint but extent > `max_extent` (1.0°) |
| `below_pt_thresh` | Extent OK but flux < point-source sensitivity at that position |
| `only_ext_penalty` | Would pass as point source, but extended-source correction raises the bar above the flux |
| `detected` | Passes all cuts |

**Usage:**
```python
breakdown = survey.get_detection_breakdown(
    population["coordinate"],
    population["flux"],
    population["extent"],
)
for label, mask in breakdown.items():
    print(f"{label:20s}: {mask.sum()} ({100*mask.mean():.1f}%)")
```

**Why it matters:** Directly answers whether detection loss is dominated by:
- Sources being outside the footprint
- Faint fluxes
- The extended-source PSF penalty

---

## Summary Table

| Change | Original | Modified | Purpose |
|--------|----------|----------|---------|
| `sensitivity_scale` | Not available | Multiplier (default 1.0) | Calibrate flux scale |
| `detection_psf` | Hardcoded `self.psf` | Separate tunable parameter | Test PSF sensitivity |
| `get_detection_breakdown()` | Not available | Returns failure categories | Diagnose non-detections |
