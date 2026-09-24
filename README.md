# Thermal-Edge-Fire: Onboard 6U CubeSat Wildfire Intelligence

Autonomous edge-computing pipeline for real-time wildfire detection and data prioritization on a 6U CubeSat carrying an uncooled Long-Wave Infrared (LWIR) microbolometer payload.

Anchored to documented flight precedent (**KITSUNE** CM3+ payload & **Chatar et al.** downlink-prioritization pipeline), enhanced with a **Dual-Gate Validation Architecture** (validation before and after treatment) to guarantee input data validity, preserve detection recall, and protect edge accelerator power budgets.

> 👥 **Team Work Division & Architecture Blueprint:** See [`docs/TEAM_ROLES_AND_TASKS.md`](docs/TEAM_ROLES_AND_TASKS.md) for the complete 3-person task breakdown across Stages 1 to 4 with strict interface contracts.

---

## 🏛️ Pipeline & Dual-Gate Validation Architecture

```
Raw Frame ──► [ GATE 1: Input Quality Gate ] ──► NUC / Calibration / Norm ──► CNN Segmentation ──► [ GATE 2: Output Sanity Gate ] ──► Downlink Queue
                    (BEFORE Treatment)                                                                      (AFTER Treatment)
```

### 1. Gate 1: Pre-Treatment Input Quality Gate (`src/fireedge/validation/`)
Runs on the **raw detector frame** before preprocessing or CNN inference to save compute:
* **Ground Mode (`ground_gate_validator.py`):** Comprehensive archive-level screening for multi-band satellite data (checks dead sensors, bit-depth, NaN/Inf, nodata bounds, and saturation excess).
* **Flight Mode (`flight_gate_validator.py`):** Ultra-fast (< 25 ms on Raspberry Pi Cortex-A72) single-channel LWIR microbolometer validator. Differentiates single-pixel radiation transients (Single-Event Upsets / Cosmic Rays) from true optical hotspots via **Point Spread Function (PSF) Spatial Halo verification** (van Dokkum 2001; Zhukov et al. 2006).

### 2. Preprocessing & Sensor Conditioning (`src/fireedge/preprocessing/`)
* **Two-Point NUC (`nuc.py`):** Restores detector uniformity across pixel gain/offset variations and replaces dead/unresponsive pixels using local neighborhood median filtering.
* **Microbolometer Degrader (`nuc.py`):** Injects realistic fixed-pattern noise (FPN), 1/f temporal drift, read noise, and dead pixels for robust CNN domain adaptation.
* **Radiometric Calibration (`calibration.py`):** Inverts Landsat-8 TIRS Planck function ($K_1, K_2, M_L, A_L$) to compute physical Brightness Temperature in Kelvin ($T_B$).
* **Normalization (`normalization.py`):** Normalizes Kelvin brightness temperatures to the $[0.0, 1.0]$ float32 range via physical window clipping or robust percentile scaling.

### 3. Edge Data Engine & Contracts (`src/fireedge/data/`)
* **`ActiveFireDataset`:** PyTorch Dataset yielding calibrated thermal image tensors and binary fire masks.
* **`ActiveFireDataModule` / `get_pan_african_dataloaders`:** Scene-level leak-free data splitting and stratified batch sampling balancing active savanna fires with North African bare soil hard negatives.
* **Person 2 Contract:** Guaranteed output tensor shapes:
  - Input Images: `[B, 1, 256, 256]`, `torch.float32`, normalized to $[0.0, 1.0]$.
  - Ground-Truth Masks: `[B, 1, 256, 256]`, `torch.float32`, binary values $\{0.0, 1.0\}$.

### 4. Gate 2: Post-Treatment Output & Physical Sanity Gate (`output_validator.py`)
Runs on the **preprocessed tensor and CNN prediction output** before queuing for downlink:
* **Thermal Contrast Verification:** Verifies physical reality:
  $$\text{Thermal Contrast Ratio} = \frac{\mu_{\text{fire}}}{\mu_{\text{background}}} \ge 1.15$$
  Suppresses cold-surface false alarms (cloud edges, snow, coastal boundaries).
* **Spatial Plausibility:** Rejects isolated single-pixel false-alarm speckles and catastrophic model collapse (> 50% frame fire hallucinations).
* **Telemetry Alert Protocol:** Formats compact CCSDS alert packets (< 100 bytes) containing fire cluster coordinates, intensity, and checksum for direct downlink.

---

## 🛰️ Tri-Dataset Strategy: Roles & Pipeline Placement

| Dataset | Designated Role & Pipeline Placement | Decisive Technical Justification | Current Status |
|---|---|---|---|
| **ActiveFire**<br>*(Pereira et al., Landsat-8)* | **PRIMARY STAGE 1 & 2 ENGINE**<br>• Core model training backbone.<br>• Semantic segmentation benchmarking.<br>• Ground-truth mask consensus (Schroeder, Kumar-Roy, Murphy, Voting). | • Native LWIR bands (B10: 10.9 µm, B11: 12.0 µm) matching uncooled microbolometer physics.<br>• $256 \times 256$ tiled patches matching CubeSat memory/SRAM constraints.<br>• 218 active patches on disk + 882 multi-algorithm consensus masks + 9,044 expert-annotated benchmark archive. | **ACTIVE & FUNCTIONAL**<br>882 masks & 218 patches indexed and batched into PyTorch tensors. |
| **TS-SatFire**<br>*(Zhao et al., Nature Sci Data 2025)* | **TEMPORAL & CROSS-SENSOR VALIDATION**<br>• Evaluates model domain adaptation from Landsat-8 to VIIRS 375 m coarse GSD.<br>• Tests fire detection across day vs. night orbital passes.<br>• Benchmark against time-series fire progressions. | • VIIRS 375 m I-4 (3.74 µm) & I-5 (11.45 µm) closely resemble coarse CubeSat uncooled thermal camera ground sampling distances (50 m–300 m).<br>• Provides real dynamic fire spread sequences over time. | **INTEGRATED & TESTED**<br>Loader in `src/fireedge/io/tssatfire.py`, acquisition script in `scripts/01_get_tssatfire.py`, tested in `tests/test_io_tssatfire.py`. |
| **THRawS / PyRawS**<br>*(ESA-PhiLab, Sentinel-2)* | **HARDWARE-IN-THE-LOOP (HIL) L0 REALISM**<br>• Tests Gate 1 on authentic raw uncompressed satellite instrument telemetry.<br>• Validates detector readout before geometric/radiometric correction. | • Standard datasets (Landsat/VIIRS) distribute Level-1 processed products.<br>• THRawS provides true **orbital Level-0 telemetry**, allowing verification against raw sensor striping, missing scanlines, raw ADC counts, and cosmic ray hits. | **INTEGRATED & TESTED**<br>Loader in `src/fireedge/io/thraws.py`, acquisition script in `scripts/02_get_thraws.py`, tested in `tests/test_io_thraws.py`. |

---

## ⚡ Quickstart & Execution Guide

### 1. Environment Setup
```bash
# Recommended Python version: Python 3.10+ (tested on Python 3.14)
pip install -r requirements.txt
```

### 2. Register Jupyter Kernel (for Notebooks)
To prevent Jupyter from connecting to Conda `base` or an incorrect Python version:
```bash
python -m ipykernel install --user --name python314 --display-name "Python 3.14 (thermal-edge-fire)"
```

### 3. Run Test Suite (34 Tests, 100% Passing)
Verifies Gate 1, Gate 2, NUC, Calibration, Normalization, DataLoaders, and I/O readers:
```bash
python -m pytest
```

### 4. Verify End-to-End Pipeline
Executes the full pipeline from raw frame through Gate 1, Two-Point NUC, Planck Calibration, Normalization, and PyTorch Tensor formatting:
```bash
python scripts/verify_pipeline.py
```

### 5. Run Pan-African Dataset Validation & Audit
```bash
# Ingest and validate African and global patches
python scripts/20_validate_dataset.py

# Generate markdown validation report
python scripts/21_validation_report.py
```

### 6. Interactive Exploratory Data Analysis (EDA)
Open [`notebooks/01_african_data_exploration.ipynb`](notebooks/01_african_data_exploration.ipynb) in VS Code or Jupyter:
* Make sure to select kernel: **`Python 3.14 (thermal-edge-fire)`**.
* Explores African thermal radiance, brightness temperature histograms, multi-algorithm mask overlays, and hot-soil false alarm resistance.

---

## 📁 Repository Structure
```
thermal-edge-fire/
├── configs/
│   └── config.yaml                  # Central configuration (paths, band maps, thresholds)
├── notebooks/
│   └── 01_african_data_exploration.ipynb # Interactive Pan-African EDA notebook
├── pytest.ini                       # Test configuration with pythonpath = src
├── requirements.txt                 # Exact dependencies
├── README.md                        # Documentation and execution guide
├── scripts/
│   ├── _bootstrap.py                # sys.path configuration for fireedge imports
│   ├── verify_pipeline.py           # End-to-end integration test (Raw -> Gate 1 -> NUC -> Tensor)
│   ├── 00_get_activefire.py         # Download & unpack ActiveFire (manual & subset releases)
│   ├── 01_get_tssatfire.py          # Download & catalog TS-SatFire
│   ├── 02_get_thraws.py             # Fetch & unpack THRawS sample scenes
│   ├── 10_explore_activefire.py     # Band stats, thermal distributions, label verification
│   ├── 11_explore_tssatfire.py      # VIIRS band verification & night/day pass analysis
│   ├── 12_explore_thraws.py         # L0 raw DN telemetry analysis & artifact inspection
│   ├── 13_explore_africa.py         # Dedicated Pan-African wildfire data exploration
│   ├── 20_validate_dataset.py       # Gate 1 dataset audit across all sets
│   └── 21_validation_report.py      # Generate Data Readiness Quality Gate Report
├── src/
│   └── fireedge/
│       ├── __init__.py
│       ├── config.py                # Configuration loader & path resolution
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py           # ActiveFireDataset: PyTorch Dataset yielding [C, H, W]
│       │   └── datamodule.py        # ActiveFireDataModule & PanAfricanBatchSampler
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── calibration.py       # Planck radiometric inversion (DN -> Kelvin)
│       │   ├── nuc.py               # TwoPointNUC & MicrobolometerDegrader
│       │   └── normalization.py     # Fixed window & robust percentile normalizers
│       ├── io/
│       │   ├── __init__.py
│       │   ├── activefire.py        # Landsat-8 GeoTIFF loader & mask pairing
│       │   ├── tssatfire.py         # TS-SatFire VIIRS reader & event organizer
│       │   └── thraws.py            # THRawS raw L0 reader & PyRawS bridge
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── base_validator.py    # ValidatorConfig, ValidationReport, base math
│       │   ├── ground_gate_validator.py # Ground Gate 1: Comprehensive archive-level screening
│       │   ├── flight_gate_validator.py # Flight Gate 1: Fast onboard microbolometer PSF halo screening
│       │   └── output_validator.py  # Gate 2: Post-treatment physical sanity & alert generator
│       └── exploration/
│           ├── __init__.py
│           ├── band_stats.py        # Quantile, noise, and SNR accumulator
│           └── separability.py      # Fisher ratio, AUC, KS tests for thermal channels
├── tests/
│   ├── conftest.py                  # Synthetic multi-band GeoTIFF generators & mock frames
│   ├── test_data_dataset.py         # PyTorch Dataset and DataLoader tests
│   ├── test_preprocessing.py        # NUC, Degrader, and Normalization tests
│   ├── test_frame_validator.py      # Base Gate 1 fault injection tests
│   ├── test_flight_validator.py     # Flight Gate 1 PSF halo & SEU tests
│   ├── test_output_validator.py     # Gate 2 physical consistency & alert tests
│   ├── test_io_activefire.py        # ActiveFire patch indexing and pairing tests
│   ├── test_io_tssatfire.py         # TS-SatFire VIIRS reader tests
│   └── test_io_thraws.py            # THRawS raw telemetry reading tests
├── reports/
│   ├── figures/                     # Generated charts, histograms, and correlation matrices
│   ├── africa_inventory.csv         # Curated 44 African patch metadata
│   ├── activefire_inventory.csv     # Curated 217 global patch metadata
│   └── data_readiness_report.md     # Comprehensive Stage 0/1 audit document
└── data/
    ├── raw/                         # Raw downloads (ActiveFire patches & masks)
    ├── interim/                     # Calibrated validator configs & metrics
    └── processed/                   # Edge-ready splits
```

---

## 🔬 Flight Precedent & Edge SBC Constraints

* **Platform Target:** 6U CubeSat with uncooled Long-Wave Infrared (LWIR) microbolometer (e.g., FLIR Lepton 3.5 / ULIS Pico384).
* **Onboard Computer:** Embedded ARM Cortex-A72 SBC (Raspberry Pi Compute Module CM3+/CM4 or similar space-qualified equivalent) running Linux.
* **Flight Pipeline Latency:**
  - Flight Gate 1 (PSF Halo check): $\approx 25\text{ ms}$
  - Calibration + Normalization: $\approx 4\text{ ms}$
  - Quantized Edge CNN (ONNX / INT8): $\approx 50\text{–}80\text{ ms}$
  - Gate 2 + CCSDS Telemetry: $\approx 2\text{ ms}$
  - **Total Latency:** $< 120\text{ ms}$ per frame (leaves CPU $> 95\%$ idle given a 2–5 second frame acquisition interval).
