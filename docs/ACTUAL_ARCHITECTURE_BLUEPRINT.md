# Architecture Blueprint: Data Engine & Preprocessing Pipeline

> **Complete Technical Reference** — every file, every function, why it exists, and how to execute it.

---

## Table of Contents
1. [Why This Architecture?](#1-why-this-architecture)
2. [Full File Map & Module Dependencies](#2-full-file-map--module-dependencies)
3. [Module-by-Module Deep Dive](#3-module-by-module-deep-dive)
4. [Data Inventory & What You Have on Disk](#4-data-inventory--what-you-have-on-disk)
5. [Execution Guide: How to Run Everything](#5-execution-guide-how-to-run-everything)
6. [What Runs on Earth vs. What Runs on the Satellite](#6-what-runs-on-earth-vs-what-runs-on-the-satellite)
7. [How Person 2 Starts (Without Running Your Full Pipeline)](#7-how-person-2-starts-without-running-your-full-pipeline)

---

## 1. Why This Architecture?

### 1.1 The Core Problem

A 6U CubeSat with a $256 \times 256$ uncooled microbolometer acquires one thermal frame every 2–5 seconds. Each raw frame is $\approx 128\text{ KB}$ of 16-bit integers. In Low Earth Orbit (LEO), the satellite has only 5–10 minutes of ground station contact per pass at low bitrate (S-band: 1–10 Mbps). Downlinking raw imagery is prohibitively expensive in power and bandwidth.

**Solution:** Process frames **onboard** using an edge CNN to extract only fire detections ($< 200$ bytes per alert), achieving $> 99.9\%$ data compression.

### 1.2 Why the Pipeline Is Layered This Way

The architecture follows the **physical signal chain** of how photons from a wildfire become an actionable alert:

```
Wildfire Photons (8–14 μm LWIR)
         │
         ▼
┌─────────────────────────────┐
│  DETECTOR READOUT           │  Raw 16-bit DN (Digital Numbers)
│  (Microbolometer FPA)       │  = electrical response of each pixel
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  GATE 1: INPUT VALIDATION   │  "Is this frame physically valid?"
│  (flight_gate_validator.py) │  Reject dead sensors, cosmic rays, saturation
└────────────┬────────────────┘
             │ PASS
             ▼
┌─────────────────────────────┐
│  NUC: NON-UNIFORMITY CORR.  │  Fix pixel-to-pixel gain/offset variations
│  (nuc.py)                   │  Repair dead pixels via local median
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  CALIBRATION                │  Convert unitless DN → physical Kelvin
│  (calibration.py)           │  via Planck function inversion
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  NORMALIZATION              │  Scale Kelvin temperatures → [0.0, 1.0]
│  (normalization.py)         │  for neural network input compatibility
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  PYTORCH TENSOR CONTRACT    │  [B, 1, 256, 256] float32 ∈ [0.0, 1.0]
│  (dataset.py / datamodule)  │  + Binary fire mask [B, 1, 256, 256]
└────────────┬────────────────┘
             │
             ▼
       Person 2's CNN
```

Each layer solves **one physical problem** and nothing else. This means:
- Any layer can be unit-tested independently.
- Any layer can be replaced without affecting the others.
- The same code runs in two contexts: **on Earth** (training) and **in orbit** (inference).

### 1.3 Why Each Module Exists

| Module | Physical Problem It Solves | What Happens If You Skip It |
| :--- | :--- | :--- |
| **Gate 1** (`flight_gate_validator.py`) | Rejects corrupted sensor readouts before wasting compute | CNN runs inference on garbage frames → false alerts → wasted downlink |
| **NUC** (`nuc.py`) | Microbolometer pixels have different sensitivities; the image has column/row stripes and dead pixels | CNN learns detector artifacts instead of fire signatures → zero generalization to real orbit |
| **Calibration** (`calibration.py`) | Raw DN integers have no physical meaning; same DN value means different temperatures on different detectors | Model trained on DN values is detector-specific and cannot transfer to flight hardware |
| **Normalization** (`normalization.py`) | Neural networks expect inputs in $[0.0, 1.0]$; temperatures range from $200\text{ K}$ to $500\text{ K}$ | Gradient explosion during training; unstable convergence; poor precision |
| **Dataset** (`dataset.py`) | Need to pair images with ground-truth fire masks and apply synchronized augmentations | No supervised learning possible without paired (image, label) tensors |
| **DataModule** (`datamodule.py`) | Need to prevent spatial data leakage and balance fire vs. non-fire samples | Overfit to specific scenes; model memorizes backgrounds instead of learning fire physics |

---

## 2. Full File Map & Module Dependencies

```
src/fireedge/                              ← Python package root
├── __init__.py                            ← Package marker
├── config.py                              ← Project-wide path resolver (reads configs/config.yaml)
│
├── io/                                    ← RAW DATA READERS (one per dataset)
│   ├── __init__.py                        ← Re-exports: read_image, read_mask, index_patches
│   ├── activefire.py                      ← Landsat-8 GeoTIFF reader + mask pairing + geo classification
│   ├── tssatfire.py                       ← VIIRS TS-SatFire reader with memory-safe decimation
│   └── thraws.py                          ← ESA-PhiLab THRawS Level-0 raw telemetry reader
│
├── preprocessing/                         ← SIGNAL CONDITIONING CHAIN
│   ├── __init__.py                        ← Re-exports: dn_to_temperature, TwoPointNUC, etc.
│   ├── calibration.py                     ← Planck inversion: DN → Radiance → Kelvin
│   ├── nuc.py                             ← Two-point NUC + Microbolometer degradation simulator
│   └── normalization.py                   ← Physical window / percentile / minmax normalizers
│
├── data/                                  ← PYTORCH DATA ENGINE
│   ├── __init__.py                        ← Re-exports: ActiveFireDataset, get_pan_african_dataloaders
│   ├── dataset.py                         ← ActiveFireDataset: reads GeoTIFFs, calibrates, normalizes, augments
│   └── datamodule.py                      ← Scene-level splitting, stratified sampling, DataLoaders
│
├── validation/                            ← DUAL-GATE QUALITY CONTROL
│   ├── __init__.py                        ← Re-exports validators
│   ├── base_validator.py                  ← ValidatorConfig, ValidationReport, shared math
│   ├── ground_gate_validator.py           ← Ground Gate 1: comprehensive archive-level multi-band audit
│   ├── flight_gate_validator.py           ← Flight Gate 1: fast single-channel PSF halo check (< 25 ms)
│   ├── frame_validator.py                 ← Legacy/base frame validator
│   └── output_validator.py                ← Gate 2: post-inference thermal consistency & spatial plausibility
│
└── exploration/                           ← EDA STATISTICS & ANALYSIS
    ├── __init__.py                        ← Re-exports analysis functions
    ├── band_stats.py                      ← Multi-band quantile, SNR, and correlation statistics
    └── separability.py                     ← Fisher ratio, KS tests for fire vs. background channels


scripts/                                   ← EXECUTABLE SCRIPTS (run from project root)
├── _bootstrap.py                          ← Adds src/ to sys.path for all scripts
├── 00_get_activefire.py                   ← Downloads ActiveFire patches from Google Drive
├── 01_get_tssatfire.py                    ← Downloads TS-SatFire from Kaggle
├── 02_get_thraws.py                       ← Downloads THRawS from ESA-PhiLab
├── 10_explore_activefire.py               ← Band statistics and thermal distributions
├── 11_explore_tssatfire.py                ← VIIRS band verification and night/day analysis
├── 12_explore_thraws.py                   ← Level-0 raw telemetry artifact inspection
├── 13_explore_africa.py                   ← Pan-African dedicated wildfire EDA
├── 20_validate_dataset.py                 ← Gate 1 dataset audit across all patches
├── 21_validation_report.py                ← Generate markdown Data Readiness Report
├── generate_eda_notebook.py               ← Script that generated the EDA notebook
└── verify_pipeline.py                     ← End-to-end integration test (7 stages)


tests/                                     ← AUTOMATED UNIT TESTS (34 tests, all passing)
├── conftest.py                            ← Synthetic GeoTIFF generators & mock frame fixtures
├── test_data_dataset.py                   ← ActiveFireDataset tensor shapes & ranges
├── test_preprocessing.py                  ← Calibration roundtrip, NUC variance reduction, normalization bounds
├── test_frame_validator.py                ← Base Gate 1 fault injection (dead frame, striping, SEU)
├── test_flight_validator.py               ← Flight Gate 1 PSF halo & cosmic ray rejection
├── test_output_validator.py               ← Gate 2 thermal consistency & hallucination rejection
├── test_io_activefire.py                  ← Patch indexing, mask pairing, scene metadata parsing
├── test_io_tssatfire.py                   ← VIIRS metadata extraction and decimated reading
└── test_io_thraws.py                      ← THRawS raw telemetry reading


notebooks/
└── 01_african_data_exploration.ipynb      ← Interactive EDA (Jupyter, kernel: Python 3.14)
```

### Dependency Flow

```
config.py ──────────────────────────────────────────────────────────┐
     │                                                              │
     ▼                                                              │
io/activefire.py ──► io/tssatfire.py ──► io/thraws.py              │
     │                                                              │
     ▼                                                              │
preprocessing/calibration.py ──► preprocessing/nuc.py              │
     │                               │                              │
     ▼                               │                              │
preprocessing/normalization.py ◄─────┘                              │
     │                                                              │
     ▼                                                              │
data/dataset.py  (uses: io + preprocessing)                        │
     │                                                              │
     ▼                                                              │
data/datamodule.py  (uses: dataset.py)                             │
     │                                                              │
     ▼                                                              │
[Person 2's CNN Model]  ◄──── INTERFACE CONTRACT                   │
     │                                                              │
     ▼                                                              │
validation/output_validator.py  ──► [Person 3's Telemetry]         │
                                                                    │
validation/flight_gate_validator.py ◄───────────────────────────────┘
```

---

## 3. Module-by-Module Deep Dive

### 3.1 `src/fireedge/config.py` — Project Configuration

**Purpose:** Centralized path resolution. All other modules call `config.py` to find data directories, report directories, and YAML configuration without hardcoded paths.

**Key Functions:**
| Function | What It Does |
| :--- | :--- |
| `get_project_root()` | Returns absolute path to `thermal-edge-fire/` by walking up from `config.py`'s location |
| `load_config()` | Reads `configs/config.yaml` (band constants, data paths, threshold presets) |
| `resolve_path(rel)` | Converts relative paths to absolute paths anchored to project root |
| `get_data_dir(key)` | Returns the path to a specific dataset root and creates it if missing |
| `get_reports_dir()` | Returns `reports/` path and ensures `reports/figures/` exists |

---

### 3.2 `src/fireedge/io/activefire.py` — Landsat-8 GeoTIFF Loader

**Purpose:** Reads the Pereira et al. ActiveFire dataset — multi-band $256 \times 256$ GeoTIFF patches and their corresponding fire masks. Pairs each image with its mask using filename normalization. Extracts Landsat scene metadata and geographic coordinates.

**Key Functions:**
| Function | What It Does |
| :--- | :--- |
| `read_image(path, channels=None)` | Reads a multi-band GeoTIFF into `np.ndarray [C, H, W]` uint16. Optional channel selection (e.g., `channels=[8]` for Band 10 only). |
| `read_mask(path, shape)` | Reads a single-band mask GeoTIFF into `np.ndarray [H, W]` uint8. Returns all-zeros if `path` is `None` (graceful fallback for patches without masks). |
| `index_patches(root, algorithm="voting")` | Scans `root/patches/` for images and `root/masks/voting/` for masks. Pairs them by normalized stem. Returns a `pd.DataFrame` with columns: `stem`, `image_path`, `mask_path`, `has_mask`, `scene_id`, `path`, `row`, `date`, `center_lat`, `center_lon`, `region`. |
| `mask_key(stem)` | Strips algorithm suffixes (e.g., `_Schroeder`, `_Murphy`, `_voting`) from filenames so that image `LC08_..._p00625.tif` can be matched to mask `LC08_..._Schroeder_p00625.tif`. |
| `parse_scene(stem)` | Extracts Landsat-8 WRS-2 Path/Row and acquisition date from the scene identifier using regex. |
| `classify_region(lat, lon)` | Assigns geographic region: `NORTH_AFRICA_MED` (27–46°N), `SUB_SAHARAN_AFRICA` (-35–27°N), or `GLOBAL_REFERENCE`. |
| `image_meta(path)` | Returns a dictionary of raster metadata: width, height, band count, CRS, bounds. |

**Why It Was Built This Way:**
- The ActiveFire dataset stores images and masks in separate directory trees with slightly different naming conventions (masks have algorithm names injected). The `mask_key()` normalization function is the glue that pairs them.
- Geographic classification (`classify_region()`) enables the Pan-African stratified sampler to guarantee hard-negative Tunisian bare soil patches appear in every training batch.

---

### 3.3 `src/fireedge/io/tssatfire.py` — VIIRS TS-SatFire Reader

**Purpose:** Reads NASA VIIRS 375 m time-series fire imagery from the TS-SatFire dataset (Zhao et al., Nature Scientific Data 2025).

**Role in Pipeline:** TS-SatFire provides **cross-sensor domain adaptation testing**. Landsat-8 has 30 m spatial resolution, but a CubeSat microbolometer has a much coarser ground sampling distance (50–300 m). VIIRS at 375 m is the closest publicly-available sensor to CubeSat resolution, so it tests whether Person 2's model generalizes to coarse imagery.

**Key Functions:**
| Function | What It Does |
| :--- | :--- |
| `list_geotiffs(root)` | Recursively finds all `.tif` and `.tiff` files under a directory. |
| `parse_date(name)` | Extracts acquisition date from a filename using regex. |
| `describe(path)` | Returns rasterio metadata dictionary (dimensions, CRS, band descriptions). |
| `read_decimated(path, max_side=512)` | Reads a large GeoTIFF with on-the-fly spatial downsampling to prevent out-of-memory errors. VIIRS scenes can be thousands of pixels wide. |

---

### 3.4 `src/fireedge/io/thraws.py` — ESA-PhiLab THRawS Reader

**Purpose:** Reads raw Level-0 satellite instrument telemetry from the ESA-PhiLab THRawS/PyRawS project.

**Role in Pipeline:** Standard datasets (Landsat, VIIRS) distribute **Level-1** products (already calibrated and geometrically corrected). In orbit, the CubeSat detector produces **Level-0** raw 16-bit integers with real noise, missing scanlines, and cosmic ray hits. THRawS provides authentic Level-0 telemetry to **stress-test Flight Gate 1** on real spacecraft sensor artifacts before the satellite flies.

**Key Functions:**
| Function | What It Does |
| :--- | :--- |
| `describe_thraws(path)` | Inspects a THRawS file (GeoTIFF or raw binary). |
| `read_raw_granule(path, channel=0)` | Reads a raw uncalibrated frame from a THRawS granule. |

---

### 3.5 `src/fireedge/preprocessing/calibration.py` — Planck Radiometric Inversion

**Purpose:** Converts raw 16-bit DN integers from the detector into physically meaningful Brightness Temperature in Kelvin using the Planck function.

**The Physics:**
```
Raw DN (Digital Number)          ← Electrical output of the detector pixel
         │
         │  L_λ = M_L × DN + A_L          (Linear radiometric rescaling)
         ▼
TOA Spectral Radiance (L_λ)     ← Energy per unit area per steradian per micron
         │
         │  T_B = K₂ / ln(K₁/L_λ + 1)    (Inverse Planck function)
         ▼
Brightness Temperature (K)       ← Physical temperature the surface would need
                                   to emit that much thermal radiation
```

**Landsat-8 TIRS Constants (hardcoded from USGS documentation):**
| Band | $M_L$ | $A_L$ | $K_1$ (W/m²·sr·μm) | $K_2$ (Kelvin) |
| :--- | :--- | :--- | :--- | :--- |
| B10 (10.9 μm) | $3.342 \times 10^{-4}$ | $0.1$ | $774.8853$ | $1321.0789$ |
| B11 (12.0 μm) | $3.342 \times 10^{-4}$ | $0.1$ | $480.8883$ | $1201.1442$ |

**Key Functions:**
| Function | Input → Output | Purpose |
| :--- | :--- | :--- |
| `dn_to_radiance(dn, band)` | `uint16 DN → float32 W/(m²·sr·μm)` | Step 1: linear rescaling |
| `radiance_to_temperature(rad, band, unit)` | `float32 radiance → float32 Kelvin` | Step 2: inverse Planck |
| `dn_to_temperature(dn, band, unit)` | `uint16 DN → float32 Kelvin` | Combined: both steps in one call |
| `temperature_to_dn(temp_k, band)` | `float32 Kelvin → uint16 DN` | **Reverse direction** — used by the sensor degradation simulator to synthesize realistic raw telemetry from known physical temperatures |

---

### 3.6 `src/fireedge/preprocessing/nuc.py` — Non-Uniformity Correction & Sensor Simulator

**Purpose:** Solves two problems:
1. **On-orbit NUC (`TwoPointNUC`):** Corrects the pixel-to-pixel gain and offset variations of the uncooled microbolometer + repairs dead/unresponsive pixels.
2. **Training domain adaptation (`MicrobolometerDegrader`):** Landsat-8 imagery is pristine space telescope data. A CubeSat microbolometer produces noisy, striped, drifting imagery. The degrader injects realistic noise so Person 2's CNN learns to be robust against these artifacts.

**Classes:**

#### `MicrobolometerConfig` (dataclass)
Configurable sensor degradation parameters. Calibrated from ULIS Pico384 / FLIR Lepton 3.5 datasheets:
| Parameter | Default | Physical Meaning |
| :--- | :--- | :--- |
| `fpn_gain_sigma` | 0.02 | 2% multiplicative gain variation across pixel array |
| `fpn_offset_sigma` | 5.0 | 5 DN additive offset spread across pixels |
| `drift_sigma` | 0.5 | Temporal 1/f drift magnitude (DN per frame) |
| `dead_pixel_frac` | 0.003 | 0.3% of pixels are stuck at 0 or saturated |
| `read_noise_sigma` | 2.0 | Per-pixel per-frame Gaussian read noise (NETD ≈ 50 mK) |
| `bit_depth` | 16 | 16-bit ADC dynamic range (0 to 65535) |

#### `TwoPointNUC`
Classical flat-field correction with bad pixel repair:

1. **Calibration** (done once with a shutter or blackbody reference):
   - Acquire a cold reference frame $C_{i,j}$ and a warm reference frame $W_{i,j}$.
   - Compute per-pixel gain: $g_{i,j} = \overline{\Delta} / (W_{i,j} - C_{i,j})$ where $\overline{\Delta} = \text{mean}(W - C)$.
   - Compute per-pixel offset: $o_{i,j} = \overline{C} - g_{i,j} \cdot C_{i,j}$ where $\overline{C} = \text{mean}(C)$.
   - Detect bad pixels: $\lvert \Delta_{i,j} - \overline{\Delta} \rvert > 3\sigma_\Delta$.

2. **Correction** (applied to every incoming raw frame):
   - $\hat{I}_{i,j} = g_{i,j} \cdot I_{i,j} + o_{i,j}$
   - Replace bad pixels with 3×3 local median filter.

#### `MicrobolometerDegrader`
Training data augmentation — injecting realistic sensor noise:
1. Fixed-pattern noise (multiplicative gain + additive offset per pixel).
2. Temporal 1/f drift (random walk accumulator).
3. Per-pixel Gaussian read noise.
4. Dead/hot pixels stuck at 0 or saturation.
5. Clip to valid detector range $[0, 2^{16} - 1]$.

---

### 3.7 `src/fireedge/preprocessing/normalization.py` — Thermal Normalizers

**Purpose:** Neural networks require inputs in $[0.0, 1.0]$. Physical temperatures span $200\text{ K}$ to $500\text{ K}$. This module maps temperatures to network-compatible ranges.

**Key Functions:**
| Function | Strategy | When to Use |
| :--- | :--- | :--- |
| `robust_percentile_norm(arr, p_min=2, p_max=98)` | Clips to 2nd–98th percentile, then linearly scales to $[0, 1]$ | **Default choice.** Resilient to extreme outliers (dead pixels, cosmic ray strikes). |
| `fixed_window_norm(temp_k, vmin=270, vmax=420)` | Linearly maps a fixed physical window ($-3°\text{C}$ to $+147°\text{C}$) to $[0, 1]$ | Use when you need absolute physical consistency: same fire temperature always maps to the same network activation across all scenes. |
| `minmax_norm(arr)` | Standard min-max scaling | Simple baseline, sensitive to outliers. |

---

### 3.8 `src/fireedge/data/dataset.py` — PyTorch ActiveFireDataset

**Purpose:** The central `torch.utils.data.Dataset` class that Person 2's training loop calls. Each `__getitem__(idx)` call:
1. Reads a raw multi-band GeoTIFF from disk.
2. Reads the corresponding binary fire mask.
3. Optionally applies microbolometer degradation (sensor noise augmentation).
4. Converts raw DN to Kelvin via Planck calibration.
5. Normalizes to $[0.0, 1.0]$.
6. Optionally applies geometric augmentations (flips, rotations, thermal jitter).
7. Returns `(image_tensor, mask_tensor)` both `torch.float32`.

**Constructor Parameters:**
| Parameter | Default | Purpose |
| :--- | :--- | :--- |
| `df` | — | Pandas DataFrame from `reports/africa_inventory.csv` |
| `channels` | `(8,)` | Band indices to extract. `8` = Landsat Band 10 (LWIR 10.9 μm) |
| `target_size` | `(256, 256)` | Output spatial dimensions (crop or pad) |
| `calibrate_temperature` | `True` | Apply Planck inversion DN→Kelvin |
| `degrader` | `None` | Optional `MicrobolometerDegrader` instance for noise augmentation |
| `norm_method` | `"robust"` | `"robust"` or `"fixed"` normalization strategy |
| `augment` | `False` | Enable random flips, rotations, thermal jitter |

---

### 3.9 `src/fireedge/data/datamodule.py` — DataLoaders & Sampling Strategy

**Purpose:** Orchestrates train/val/test splitting and batch construction.

**Key Components:**

#### `scene_level_split(df, train_frac=0.70, val_frac=0.15)`
Splits patches by their parent Landsat acquisition scene ID. Patches from the same physical scene share identical atmospheric conditions and ground cover. If they leaked across train/test splits, the model would memorize scene-specific backgrounds rather than learning fire physics.

#### `PanAfricanBatchSampler`
Stratified batch sampler ensuring every training batch contains:
- ≥50% active fire patches (Sub-Saharan savanna/forest fires → high recall training signal).
- Remaining slots filled with hard-negative bare soil patches (North Africa / Tunisia → anti-false-alarm training signal).

#### `get_pan_african_dataloaders(df, batch_size=8)`
Main API that returns `(train_loader, val_loader, test_loader)` ready for Person 2.

---

### 3.10 `src/fireedge/validation/` — Dual-Gate Quality Control

#### `ground_gate_validator.py` (Ground Gate 1)
Comprehensive batch audit for training data curation. Checks: NaN/Inf, dead sensors, bit-depth violations, nodata excess, saturation overflow, and hot pixel candidates.

#### `flight_gate_validator.py` (Flight Gate 1)
Ultra-fast ($< 25\text{ ms}$ on ARM) single-channel validator for in-orbit use. Uses **Point Spread Function (PSF) Spatial Halo verification** to distinguish real optical fire hotspots (which exhibit lens-blur halos in neighboring pixels) from cosmic ray Single-Event Upsets (which create sharp Dirac-delta spikes with zero optical diffusion).

#### `output_validator.py` (Gate 2)
Post-inference sanity check. Verifies thermal contrast ratio $\ge 1.15$ between fire and background pixels. Rejects isolated single-pixel speckles and catastrophic model collapse (>50% frame classified as fire).

---

### 3.11 `src/fireedge/exploration/` — EDA Statistics

#### `band_stats.py`
Computes per-band quantile statistics, noise estimates, and signal-to-noise ratios across the dataset.

#### `separability.py`
Computes the M-statistic (Fisher discriminant ratio) between fire and background pixel distributions to quantify how distinguishable fires are in each spectral band.

---

## 4. Data Inventory & What You Have on Disk

| Asset | Count | Path | Size | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Landsat-8 Multispectral Patches** | 218 files | `data/raw/activefire/patches/` | ~1 GB | ✅ On disk |
| **Algorithmic Fire Masks** (Schroeder, Murphy, etc.) | 882 files | `data/raw/activefire/masks/voting/` | ~200 MB | ✅ On disk |
| **African Inventory CSV** | 44 entries | `reports/africa_inventory.csv` | 5 KB | ✅ Generated |
| **Global Inventory CSV** | 217 entries | `reports/activefire_inventory.csv` | 30 KB | ✅ Generated |
| **External ActiveFire Reference Code** | Many | `data/external/activefire/` | ~50 MB | ✅ Downloaded |

### What About the Full 30 GB Dataset?

The full ActiveFire dataset (Pereira et al.) contains **9,044 manually annotated patches** across the entire globe. Your current disk has only 218 patches (African subset). To download the full archive:

```bash
python scripts/00_get_activefire.py --release manual
```

This downloads `manual_annotations_patches.zip` from Google Drive. **Person 2 does NOT need this to start.** The 44 African patches are sufficient for model prototyping and architecture validation. The full 30 GB is needed only for **final training at scale**.

---

## 5. Execution Guide: How to Run Everything

### 5.1 Prerequisites

```bash
# Verify Python 3.14 (or 3.10+)
python --version

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel (for the notebook)
python -m ipykernel install --user --name python314 --display-name "Python 3.14 (thermal-edge-fire)"
```

### 5.2 Run Order (Scripts)

| # | Command | What It Does | Time |
| :--- | :--- | :--- | :--- |
| 1 | `python -m pytest` | Run all 34 unit tests | ~4 seconds |
| 2 | `python scripts/verify_pipeline.py` | End-to-end integration test (7 stages) | ~3 seconds |
| 3 | `python scripts/10_explore_activefire.py` | Band statistics for ActiveFire patches | ~30 seconds |
| 4 | `python scripts/13_explore_africa.py` | Pan-African wildfire thermal contrast analysis | ~60 seconds |
| 5 | `python scripts/20_validate_dataset.py` | Gate 1 batch audit across all patches | ~2 minutes |
| 6 | `python scripts/21_validation_report.py` | Generate Data Readiness Report | ~30 seconds |

### 5.3 Run the EDA Notebook

1. Open `notebooks/01_african_data_exploration.ipynb` in VS Code.
2. Click the kernel selector (top right) → **"Select Another Kernel"** → **"Jupyter Kernel"** → **"Python 3.14 (thermal-edge-fire)"**.
3. Run All Cells.

---

## 6. What Runs on Earth vs. What Runs on the Satellite

| Component | On Earth (Ground Station) | On Satellite (Raspberry Pi) |
| :--- | :---: | :---: |
| **Dataset Download & Indexing** (`scripts/00_*.py`) | ✅ | ❌ |
| **EDA & Exploration** (`scripts/10_*.py`, notebook) | ✅ | ❌ |
| **Model Training** (Person 2's training loop) | ✅ (GPU workstation) | ❌ |
| **Flight Gate 1** (`flight_gate_validator.py`) | ✅ (testing) | ✅ (every frame) |
| **NUC** (`nuc.py` — `TwoPointNUC.correct()`) | ✅ (testing) | ✅ (every frame) |
| **Calibration** (`calibration.py`) | ✅ (testing) | ✅ (every frame) |
| **Normalization** (`normalization.py`) | ✅ (testing) | ✅ (every frame) |
| **CNN Inference** (ONNX model) | ❌ | ✅ (every frame) |
| **Gate 2** (`output_validator.py`) | ✅ (testing) | ✅ (every frame) |
| **CCSDS Telemetry** (Person 3) | ❌ | ✅ (only fire detections) |
| **PyTorch DataLoaders** | ✅ (training only) | ❌ |
| **MicrobolometerDegrader** | ✅ (augmentation) | ❌ |

**Key insight:** The preprocessing code (`calibration.py`, `nuc.py`, `normalization.py`) runs in **both** contexts. On Earth, it's called by `ActiveFireDataset` during training. In orbit, it's called by the flight software before passing the frame to the ONNX model.

---

## 7. How Person 2 Starts (Without Running Your Full Pipeline)

### Person 2 does NOT need to:
- ❌ Download the full 30 GB dataset.
- ❌ Run `scripts/00_get_activefire.py`.
- ❌ Run `scripts/10_explore_activefire.py` or the EDA notebook.
- ❌ Run `scripts/20_validate_dataset.py`.

### Person 2 only needs to:

**Step 1:** Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

**Step 2:** Verify the pipeline works:
```bash
python -m pytest                    # 34 tests pass
python scripts/verify_pipeline.py   # 7 stages pass
```

**Step 3:** Use the existing 44 African patches to prototype their model:
```python
import sys; sys.path.insert(0, "src")
import pandas as pd
from fireedge.data.datamodule import get_pan_african_dataloaders

# Person 1 already generated this inventory file
df = pd.read_csv("reports/africa_inventory.csv")

# Get ready-to-use DataLoaders
train_loader, val_loader, test_loader = get_pan_african_dataloaders(
    df, batch_size=8, channels=(8,), simulate_nuc=False, num_workers=0
)

# Training loop
for epoch in range(100):
    for images, masks in train_loader:
        # images: [8, 1, 256, 256], torch.float32, range [0.0, 1.0]
        # masks:  [8, 1, 256, 256], torch.float32, binary {0.0, 1.0}
        logits = model(images)
        loss = focal_tversky_loss(logits, masks)
        loss.backward()
        optimizer.step()
```

**Step 4 (later, for full-scale training):** Download the full 30 GB manual annotations:
```bash
python scripts/00_get_activefire.py --release manual
python scripts/13_explore_africa.py   # Regenerates inventory with new patches
```
Then re-read the updated `reports/africa_inventory.csv` into the DataLoader.

> **Bottom line:** Person 2 can start designing and training their CNN model **right now** using the data that's already on disk. The DataLoader handles all calibration, normalization, and augmentation automatically. Person 2 never touches raw GeoTIFFs directly.
