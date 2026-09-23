# Thermal-Edge-Fire: Onboard 6U CubeSat Wildfire Intelligence

Autonomous edge-computing pipeline for real-time wildfire detection and data prioritization on a 6U CubeSat carrying an uncooled Long-Wave Infrared (LWIR) microbolometer payload.

Anchored to documented flight precedent (**KITSUNE** CM3+ payload & **Chatar et al.** downlink-prioritization pipeline), enhanced with a **Dual-Gate Validation Architecture** (validation before and after treatment) to guarantee input data validity and protect detection recall (§6.3, in response to industry feedback from John McDonald, IEEE Life Fellow).

> 👥 **Team Work Division & Architecture Blueprint:** See [`docs/TEAM_ROLES_AND_TASKS.md`](docs/TEAM_ROLES_AND_TASKS.md) for the complete 3-person task breakdown across Stages 1 to 4 with strict interface contracts.

---

## 🏛️ Pipeline & Dual-Gate Validation Architecture

```
Raw Frame ──► [ GATE 1: Input Quality Gate ] ──► NUC / Norm ──► CNN Segmentation ──► [ GATE 2: Output Sanity Gate ] ──► Downlink Queue
                    (BEFORE Treatment)                                                       (AFTER Treatment)
```

### 1. Gate 1: Pre-Treatment Input Quality Gate (`frame_validator.py`)
Runs on the **raw detector frame** before any pre-processing or CNN inference:
* **Detector Health:** Rejects stuck/dead sensor frames (`DEAD_OR_STUCK`), massive saturation (`SATURATION_EXCESS` > 2%), and telemetry dropouts (`NODATA_EXCESS` > 30%).
* **Readout Integrity:** Detects defective column/row striping via median profile residuals.
* **Recall-Preservation Logic:** Distinguishes radiation transients (Single-Event Upsets) from real fire candidates via dual-band confirmation across **Band 10 (10.9 µm) and Band 11 (12.0 µm)**.

### 2. Gate 2: Post-Treatment Output & Physical Sanity Gate (`output_validator.py`)
Runs on the **preprocessed tensor and CNN prediction output** before queuing for downlink:
* **Preprocessing Sanity:** Rejects calibration breakdown producing NaNs, infinities, or zero variance.
* **Thermal Consistency Verification:** Physical reality check on CNN predictions:
  $$\text{Thermal Contrast Ratio} = \frac{\mu_{\text{fire}}}{\mu_{\text{background}}} \ge 1.15$$
  If the CNN classifies a cold surface (e.g., cloud edges, snow, coastal boundaries) as fire, Gate 2 suppresses the false alarm.
* **Spatial Plausibility:** Rejects isolated 1-pixel false-alarm speckles and catastrophic model collapse (> 50% frame fire hallucinations).
* **Packet Protocol Integrity:** Validates the ~200-byte alert packet schema, coordinates within orbital swath, and checksum.

---

## 🛰️ Tri-Dataset Strategy & Roles

| Dataset | Designated Role | Decisive Technical Justification | Limitations & Citing Strategy |
|---|---|---|---|
| **ActiveFire**<br>*(Pereira et al., Landsat-8)* | **PRIMARY**<br>(Model training & segmentation benchmark) | • Native LWIR bands (B10: 10.9 µm, B11: 12.0 µm) matching uncooled microbolometer physics.<br>• 256×256 tiled patches matching CubeSat memory/SRAM constraints.<br>• 9,044 human-expert annotated ground-truth patches + 146,214 global scenes.<br>• Spatial resolution (30 m resampled / 100 m native) isolates sub-hectare hotspots. | Resampled L1T data (calibrated). Simulated raw degradation models edge detector noise. |
| **THRawS / PyRawS**<br>*(ESA-PhiLab, Sentinel-2)* | **ONBOARD REALISM**<br>(L0 Quality Gate & domain-shift test) | • **Authentic orbital Level-0 raw data** directly from ESA instrument telemetry.<br>• Tests Gate 1 on **real spacecraft sensor artifacts** (detector striping, missing scanlines, raw integer ADC counts, cosmic ray single-event upsets).<br>• PyRawS provides Pythonic decompression and calibration tools. | Uses Sentinel-2 SWIR (B11/B12, ~1.6/2.2 µm), not LWIR. Used to prove raw telemetry ingestion and domain-shift survival. |
| **TS-SatFire**<br>*(Zhao et al., Nature Sci Data 2025)* | **SECONDARY**<br>(Diurnal/night context & baseline contrast) | • Verified VIIRS 375 m I4 (MWIR 3.74 µm) & I5 (LWIR 11.45 µm) time series across 179 fire events.<br>• Tests behavior across day vs. night passes.<br>• Contrasts lightweight single-frame edge inference against heavy ground-based temporal models. | At 375 m/pixel, each pixel is 14 hectares. Requires multi-day time series buffering impossible on a 6U edge SBC. |

---

## ⚡ Quickstart & Execution Guide

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite (23 Tests, Zero External Dependencies)
Verifies both Gate 1 and Gate 2 software logic with synthetic fault-injection frames:
```bash
python -m pytest tests/ -v
```

### 3. Data Acquisition
```bash
# Pereira et al. ActiveFire (Primary Training: 9,044 manual patches)
python scripts/00_get_activefire.py --release manual

# TS-SatFire VIIRS (Secondary Context: Nature Sci Data 2025)
python scripts/01_get_tssatfire.py

# ESA-PhiLab THRawS (Level-0 Raw Realism)
python scripts/02_get_thraws.py
```

### 4. Data Exploration & Thermal Separability
```bash
# Explore ActiveFire thermal vs SWIR distributions, correlations, and Fisher separability
python scripts/10_explore_activefire.py

# Explore TS-SatFire VIIRS bands and event catalog
python scripts/11_explore_tssatfire.py

# Explore THRawS Level-0 raw instrument telemetry and striping profiles
python scripts/12_explore_thraws.py
```

### 5. Dual-Gate Calibration & Quality Gate Audit
```bash
# Calibrate thresholds on clean frames and measure false rejection rate on fire (< 1%)
python scripts/20_validate_dataset.py

# Compile Stage 0/1 Data Readiness Report
python scripts/21_validation_report.py
```

---

## 📁 Repository Structure
```
thermal-edge-fire/
├── configs/
│   └── config.yaml                  # Central configuration (paths, band maps, thresholds)
├── pytest.ini                       # Test configuration with pythonpath = src
├── requirements.txt                 # Exact dependencies
├── README.md                        # Documentation and execution guide
├── scripts/
│   ├── _bootstrap.py                # sys.path configuration for fireedge imports
│   ├── 00_get_activefire.py         # Download & unpack ActiveFire (manual & subset releases)
│   ├── 01_get_tssatfire.py          # Download & catalog TS-SatFire from Kaggle
│   ├── 02_get_thraws.py            # Fetch & unpack THRawS sample scenes via PyRawS
│   ├── 10_explore_activefire.py     # Band stats, thermal distributions, label verification
│   ├── 11_explore_tssatfire.py      # VIIRS band verification & night/day pass analysis
│   ├── 12_explore_thraws.py         # L0 raw DN telemetry analysis & artifact inspection
│   ├── 20_validate_dataset.py       # Calibrate and run FrameValidator across all sets
│   └── 21_validation_report.py      # Generate Stage 0/1 Data Readiness Quality Gate Report
├── src/
│   └── fireedge/
│       ├── __init__.py
│       ├── config.py                # Configuration loader & path resolution
│       ├── io/
│       │   ├── __init__.py
│       │   ├── activefire.py        # Landsat-8 GeoTIFF loader & mask pairing
│       │   ├── tssatfire.py         # TS-SatFire VIIRS reader & event organizer
│       │   └── thraws.py            # THRawS raw L0 reader & PyRawS bridge
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── frame_validator.py   # GATE 1: Pre-treatment Input Quality Gate
│       │   └── output_validator.py  # GATE 2: Post-treatment Output & Physical Sanity Gate
│       └── exploration/
│           ├── __init__.py
│           ├── band_stats.py        # Quantile, noise, and SNR accumulator
│           └── separability.py      # Fisher ratio, AUC, KS tests for thermal channels
├── tests/
│   ├── conftest.py                  # Synthetic multi-band GeoTIFF generators & mock frames
│   ├── test_frame_validator.py      # GATE 1 fault injection tests (striping, SEU, dead frame)
│   ├── test_output_validator.py     # GATE 2 tests (thermal consistency, hallucination check)
│   ├── test_io_activefire.py        # Validation tests for Pereira patch indexing
│   ├── test_io_tssatfire.py         # Tests for VIIRS metadata and decimated reading
│   └── test_io_thraws.py            # Tests for THRawS raw telemetry reading
├── reports/
│   ├── figures/                     # Generated charts, histograms, and correlation matrices
│   └── data_readiness_report.md     # Comprehensive Stage 0/1 audit document
└── data/
    ├── raw/                         # Raw downloads (gitignored)
    ├── interim/                     # Calibrated validator configs & metrics
    └── processed/                   # Edge-ready splits
```

---

## 🔬 Scientific Rigor: Synthetic Tests vs. Empirical Findings

* **Software Correctness:** The 23 unit tests run on synthetic frames to guarantee that edge-case math, fault-injection filters, and multi-band indexing never crash the onboard runtime.
* **Empirical Validation:** All scientific findings (dynamic ranges, Fisher ratios, ROC-AUC, and the **< 1.0% False Rejection Rate on Fire**) are calculated by running the pipeline directly on real satellite data, compiled into [`reports/data_readiness_report.md`](file:///c:/Users/Dell/Desktop/GeoAI/IASTAM/thermal-edge-fire/reports/data_readiness_report.md).
