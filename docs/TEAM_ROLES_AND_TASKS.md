# 🛰️ Thermal-Edge-Fire: Team Task Division & Aerospace Engineering Blueprint
### Onboard Thermal/IR Edge-Computing for Wildfire Detection on a 6U CubeSat
**Target Context:** National Earth Observation CubeSat (Tunisia CNCT / NSAC26)  
**Engineering Philosophy:** Dual-Gate Aerospace Validation, Recall-Prioritized Intelligence, Zero-Collision Decoupled Pipeline

---

## 1. Executive Mission & System Architecture

Modern Earth-observation satellites produce far more raw data than available radio downlink bandwidth can support. Rather than treating this as a transmission bottleneck (requiring gigabit laser downlinks and massive ground networks), this project treats it as an **onboard content problem**:
> *"Most captured thermal pixels are barren ground or cold ocean. Transmit only the fire, not the frame."*

### 1.1 The Operational Flight Constraints
* **Orbital Dynamics:** A 6U CubeSat in Low Earth Orbit (LEO, ~500 km altitude) travels at **$7.6\text{ km/s}$**, traversing from northern Tunisia ($37^\circ\text{N}$) to equatorial Africa in under 8 minutes.
* **Payload Constraints:** Single-channel uncooled Long-Wave Infrared (LWIR, $8\text{--}14\,\mu\text{m}$) microbolometer detector ($256 \times 256$ or $512 \times 512$ array).
* **Compute Envelope:** Ultra-low power edge SBC (Raspberry Pi Compute Module CM3+/CM4 or similar space-qualified ARM equivalent), 2–5 W power budget. Optional accelerator: Google Coral Edge TPU, Intel Myriad X, or Hailo-8.
* **Aerospace Feedback (John McDonald, IEEE Life Fellow):**
  1. **Data Validity:** Raw input telemetry must be mathematically validated before any processing touches it.
  2. **Protect Recall:** A missed wildfire (false negative) is a catastrophic failure; false alarms can be verified, but missed fires cannot be recovered.
  3. **Downlink Telemetry Resilience:** Deliver lightweight ~200-byte alert packets into a persistent, retriable priority queue that survives interrupted ground station passes.

---

## 2. Horizontal Pipeline Decoupling: How 3 People Work Without Waiting

To eliminate merge conflicts, duplicate work, and blocking dependencies, the project is divided **horizontally across pipeline stages** rather than geographically:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END ORBITAL DATAFLOW PIPELINE                            │
└────────────────────────────────────────────────────────────────────────────────────────┘

    [ Raw Sensor Telemetry / Landsat-8 TIRS / TS-SatFire ]
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 1: PRE-TREATMENT INPUT QUALITY GATE                │ ◄── PERSON 1 ✅ COMPLETE
│ • In-Flight Sensor Validation (Dead sensor, saturation)  │
│ • Mode A: Dual-band B10/B11 ground truth curation        │
│ • Mode B: Single-band PSF spatial halo (SEU rejection)   │
└─────────────────────────────┬────────────────────────────┘
                              │ PASS / PASS_WITH_FLAGS
                              ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: SENSOR PRE-PROCESSING & DATA ENGINE             │ ◄── PERSON 1 ✅ COMPLETE
│ • Radiometric Calibration (DN ──> Kelvin / Celsius)      │
│ • Microbolometer NUC & Sensor Degradation Simulator      │
│ • Dynamic Thermal Normalization [0.0, 1.0]               │
│ • Pan-African PyTorch ActiveFireDataset & DataLoader     │
│ • EDA Notebook & Dataset Inventory & Validation          │
└─────────────────────────────┬────────────────────────────┘
                              │ images: Tensor [B, 1, 256, 256]
                              │ masks:  Tensor [B, 1, 256, 256]
                              ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 3: EDGE NEURAL NETWORK MODEL & INFERENCE           │ ◄── PERSON 2
│ • Lightweight 1-Channel LWIR CNN (Tiny-UNet / MobileNet) │
│ • Class Imbalance Losses (Focal Tversky / Soft-Dice)     │
│ • Post-Training Quantization (INT8) & ONNX Export        │
└─────────────────────────────┬────────────────────────────┘
                              │ fire_probability_map: [H, W]
                              ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 4: POST-TREATMENT QUALITY GATE & ALERT DOWNLINK    │ ◄── PERSON 3
│ • Gate 2 Output Sanity: Thermal elevation check (ΔT≥10K) │
│ • Spatial plausibility filter (anti-hallucination)       │
│ • Compact ~200-byte CCSDS Alert Packet Generator         │
│ • Retriable Downlink Priority Queue                      │
└──────────────────────────────────────────────────────────┘
```

### 2.1 Strict Interface Contracts (Zero-Wait Protocol)

| Between | Interface Contract | Format / Shape | Purpose |
| :--- | :--- | :--- | :--- |
| **Person 1 $\to$ Person 2** | `train_loader`, `val_loader` | `images`: `torch.float32 [B, 1, 256, 256]`<br>`masks`: `torch.float32 [B, 1, 256, 256]` | Standard normalized thermal batch for CNN training. |
| **Person 2 $\to$ Person 3** | `infer(frame)` & `model.onnx` | Input: `np.float32 [1, H, W]`<br>Output: `np.float32 [H, W]` $\in [0.0, 1.0]$ | Pixel-level fire probability mask delivered to post-processing. |
| **Person 3 $\to$ Orbit Radio** | `CCSDS_Alert_Packet` | Binary struct (~200 bytes) | Alert payload with centroid coordinates, confidence, and CRC32 checksum. |

---

## 3. Person 1: In-Flight Sensor Validation, Preprocessing & Data Engine (Stages 1 & 2) — ✅ COMPLETE
> **Assigned to:** **You (Person 1)**  
> **Status:** **ALL TASKS COMPLETE — 34/34 unit tests passing, end-to-end pipeline verified.**  
> **Core Focus:** In-flight sensor validation (Gate 1), ground truth dual-band curation, physical thermal calibration, realistic microbolometer noise simulation, and high-performance PyTorch loaders.  
> **Modules Owned:** `src/fireedge/validation/` (Gate 1: `flight_gate_validator.py`, `frame_validator.py`, `base_validator.py`), `src/fireedge/preprocessing/`, `src/fireedge/data/`, `src/fireedge/io/`, `src/fireedge/exploration/`

### 3.1 Completed Deliverables for Person 1

#### ✅ Task 1.0: Gate 1 In-Flight Sensor Validation & Ground Curation (`src/fireedge/validation/`)
* **Flight Mode (`FlightGateValidator`):** In-orbit pre-treatment quality gate running on raw single-band LWIR microbolometer frames *before* pre-processing or CNN inference.
  - Checks dead / stuck detector lines, excessive non-finite values, saturation limits, and line striping.
  - **PSF Spatial Halo Ratio (van Dokkum 2001; Zhukov 2006):** Verifies the point-spread optical halo elevation to differentiate single-event upset (SEU) cosmic radiation hits from real sub-pixel optical fires.
* **Ground Mode (`GroundGateValidator`):** Cross-checks hot pixels across dual thermal channels (Band 10 vs. Band 11) for high-integrity dataset curation.
* **Configuration:** `ValidatorConfig` with bit-depth, radiometric noise thresholds, saturation levels, and `halo_min_ratio`.
* **Tests:** `tests/test_flight_validator.py` (4/4 tests), `tests/test_frame_validator.py` (8/8 tests) — verified SEU rejection, saturation flags, striping detection, and shape contracts.

#### ✅ Task 1.1: Radiometric Calibration (`src/fireedge/preprocessing/calibration.py`)
* **Implemented:** Planck law inversion for raw DN $\to$ Brightness Temperature in Kelvin.
  $$L_\lambda = M_L \cdot \text{DN} + A_L \qquad T_B = \frac{K_2}{\ln\left(\frac{K_1}{L_\lambda} + 1\right)}$$
* **Key Functions:** `dn_to_radiance()`, `radiance_to_temperature()`, `dn_to_temperature()`, `temperature_to_dn()`.
* **Tests:** `tests/test_preprocessing.py` — verified DN $\leftrightarrow$ Kelvin roundtrip and physical ambient range.

#### ✅ Task 1.2: Microbolometer Degradation & NUC Simulator (`src/fireedge/preprocessing/nuc.py`)
* **Implemented:** `TwoPointNUC` (classical two-point flat-field correction with bad pixel detection and replacement via local median filter), `MicrobolometerDegrader` (fixed-pattern noise, 1/f temporal drift, read noise, dead pixel injection), and `MicrobolometerConfig` (configurable sensor parameters).
* **Tests:** `tests/test_preprocessing.py::test_microbolometer_degrader_and_nuc` — verified FPN injection raises variance, NUC correction substantially restores uniformity.

#### ✅ Task 1.3: Dynamic Thermal Normalization (`src/fireedge/preprocessing/normalization.py`)
* **Implemented:** `robust_percentile_norm()` (outlier-resistant), `fixed_window_norm()` (physical 270–420 K window), `minmax_norm()`.
* **Tests:** `tests/test_preprocessing.py::test_thermal_normalization_bounds` — verified output strictly in $[0.0, 1.0]$.

#### ✅ Task 1.4: PyTorch ActiveFireDataset (`src/fireedge/data/dataset.py`)
* **Implemented:** Full `torch.utils.data.Dataset` subclass with on-the-fly calibration, optional degradation, normalization, size adjustment, and thermal augmentations (flips, rotations, contrast jitter).
* **Tests:** `tests/test_data_dataset.py` — verified tensor shapes `[1, 256, 256]`, dtype `float32`, range `[0.0, 1.0]`, binary mask values `{0.0, 1.0}`.

#### ✅ Task 1.5: Pan-African DataLoader & Hard-Negative Mining (`src/fireedge/data/datamodule.py`)
* **Implemented:** Scene-level spatial splitting (prevents data leakage), `PanAfricanBatchSampler` (guarantees ≥50% fire patches + hard-negative bare soil per batch), `get_pan_african_dataloaders()`.
* **Tests:** `tests/test_data_dataset.py` — verified zero scene-ID leakage between train/val/test splits.

#### ✅ Task 1.6: I/O Loaders for All Three Datasets
* **ActiveFire:** `src/fireedge/io/activefire.py` — GeoTIFF reader, mask pairing, scene metadata extraction, geographic region classification (tested in `tests/test_io_activefire.py`).
* **TS-SatFire:** `src/fireedge/io/tssatfire.py` — VIIRS GeoTIFF reader with memory-safe decimation (tested in `tests/test_io_tssatfire.py`).
* **THRawS:** `src/fireedge/io/thraws.py` — Raw Level-0 telemetry reader with PyRawS bridge (tested in `tests/test_io_thraws.py`).

#### ✅ Task 1.7: Exploration & EDA
* **Implemented:** `src/fireedge/exploration/band_stats.py` (multi-band quantile and SNR statistics), `src/fireedge/exploration/separability.py` (Fisher ratio, KS tests for thermal channels).
* **Notebook:** `notebooks/01_african_data_exploration.ipynb` — interactive Pan-African EDA with thermal histograms, mask overlays, and hard-negative analysis.

#### ✅ Task 1.8: Gate 1 Ground Validation & Pipeline Verification
* **Implemented:** `scripts/20_validate_dataset.py`, `scripts/21_validation_report.py`, `scripts/verify_pipeline.py`.
* **Result:** All 7 pipeline stages verified end-to-end. Gate 1 passes valid frames. All 34 tests pass.

### 3.2 Data Inventory Produced by Person 1

| Asset | Count | Location | Status |
| :--- | :--- | :--- | :--- |
| Landsat-8 GeoTIFF Patches | 218 | `data/raw/activefire/patches/` | ✅ On disk |
| Multi-Algorithm Consensus Fire Masks | 882 | `data/raw/activefire/masks/voting/` | ✅ On disk |
| African Subset Inventory | 44 entries | `reports/africa_inventory.csv` | ✅ Generated |
| Global Patch Inventory | 217 entries | `reports/activefire_inventory.csv` | ✅ Generated |

---

## 4. Person 2: Edge Neural Network & Quantization (Stage 3)
> **Core Focus:** Designing ultra-lightweight thermal segmentation CNNs, training with fire-imbalanced loss functions, and post-training INT8 quantization for edge SBCs.  
> **Modules Owned:** `src/fireedge/models/`, `src/fireedge/training/`, `src/fireedge/quantization/`

### 4.1 How Person 2 Can Start Immediately

> **Person 2 does NOT need to run Person 1's full 30 GB data pipeline.**

Person 1 has already:
1. Built the `ActiveFireDataset` and `get_pan_african_dataloaders()` that handle all calibration, normalization, and augmentation on-the-fly.
2. Generated `reports/africa_inventory.csv` (44 African patches, ~1 GB on disk) which is enough to begin prototyping.

**Person 2 simply needs to:**
```python
import sys; sys.path.insert(0, "src")
import pandas as pd
from fireedge.data.datamodule import get_pan_african_dataloaders

df = pd.read_csv("reports/africa_inventory.csv")
train_loader, val_loader, test_loader = get_pan_african_dataloaders(df, batch_size=8)

for images, masks in train_loader:
    # images: [8, 1, 256, 256], torch.float32, normalized [0.0, 1.0]
    # masks:  [8, 1, 256, 256], torch.float32, binary {0.0, 1.0}
    predictions = model(images)
    loss = loss_fn(predictions, masks)
    ...
```

To scale to the full 30 GB dataset later:
```bash
python scripts/00_get_activefire.py --release manual    # Downloads 9,044 expert-annotated patches
python scripts/13_explore_africa.py                      # Regenerates africa_inventory.csv with new data
```

### 4.2 Detailed Tasks for Person 2

#### Task 2.1: Lightweight Thermal CNN Backbones (`src/fireedge/models/`)
* **Objective:** Design neural architectures that fit within CubeSat edge SBC hardware constraints:
  - Input: Single-channel thermal tensor `[B, 1, 256, 256]`.
  - Output: Binary fire segmentation logits `[B, 1, 256, 256]`.
  - Budget constraints: **$<1.5\text{M}$ parameters, $<500\text{ MFLOPs}$, $<10\text{ MB}$ unquantized**.
* **Candidate Architectures to Implement:**
  1. **`TinyUNet`**: Depthwise separable convolutions, 4 downsampling stages (channels: 16, 32, 64, 128), skip connections.
  2. **`MobileNetV3-Fire`**: MobileNetV3-Small backbone with lightweight Feature Pyramid Network (FPN) head.

#### Task 2.2: Extreme Imbalance Loss Formulation (`src/fireedge/training/losses.py`)
* **Objective:** Standard Binary Cross-Entropy (BCE) fails because $99.98\%$ of pixels are background (the network learns to predict all zeros and achieves $99.98\%$ accuracy while detecting 0 fires).
* **Loss Functions to Implement:**
  - **Focal Loss:** Down-weights easy background pixels:
    $$\mathcal{L}_{\text{Focal}} = -\alpha (1 - p_t)^\gamma \log(p_t)$$
  - **Tversky / Focal Tversky Loss:** Explicitly prioritizes **Recall** ($\beta > \alpha$) over precision:
    $$T(\alpha, \beta) = \frac{\sum p_i g_i + \epsilon}{\sum p_i g_i + \alpha \sum p_i (1 - g_i) + \beta \sum (1 - p_i) g_i + \epsilon}$$
    $$\mathcal{L}_{\text{FT}} = (1 - T)^\gamma \quad (\text{with } \beta = 0.7, \alpha = 0.3)$$

#### Task 2.3: Training, Validation & Metric Evaluation Loop (`src/fireedge/training/trainer.py`)
* **Objective:** Automated PyTorch training loop saving best checkpoint based on validation **Fire Recall** and **IoU**.
* **Key Deliverables:**
  - Automated mixed-precision training (`torch.cuda.amp` or CPU fallback).
  - Metric computation:
    - **Recall (Primary Metric §6.3)**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
    - **IoU (Intersection-over-Union)**: $\frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$
    - **F1-Score / Dice Coefficient**
  - Checkpoint manager saving `models/best_model.pt`.

#### Task 2.4: Post-Training Quantization (PTQ) & Edge Export (`src/fireedge/quantization/`)
* **Objective:** Prepare the model for satellite Raspberry Pi (ARM Cortex-A72) and optional accelerators.
* **Key Tasks:**
  - Export PyTorch model to ONNX: `torch.onnx.export(..., "models/fire_edge.onnx")`.
  - INT8 Post-Training Quantization (PTQ) using calibration samples: compresses model from ~6 MB to $<1.5\text{ MB}$ with $<1\%$ recall degradation.
  - Python inference wrapper: `infer(frame: np.ndarray) -> np.ndarray` (fire probability map).

### 4.3 How Person 2 Works Without Waiting for Person 1
* ✅ **Person 1 is finished.** Person 2 can directly use `get_pan_african_dataloaders()` on the existing 44 African patches to prototype, then scale up to the full 30 GB archive by running `python scripts/00_get_activefire.py --release manual`.
* Person 2 can also immediately develop models and loss functions using random synthetic tensors: `torch.randn(8, 1, 256, 256)` and `torch.randint(0, 2, (8, 1, 256, 256))`.

---

## 5. Person 3: Aerospace Output Sanity, Post-Processing & Telemetry (Gate 2 & Stage 4)
> **Core Focus:** Physical post-inference sanity checks (Gate 2), ~200-byte CCSDS alert packet generation, retriable downlink priority queue, and edge benchmarking.  
> **Modules Owned:** `src/fireedge/validation/output_validator.py`, `src/fireedge/telemetry/`, `src/fireedge/benchmarking/`  
> *(Note: Stage 1 Gate 1 In-Flight Sensor Validation has been completed and assigned to Person 1).*

### 5.1 Detailed Tasks for Person 3

#### Task 3.1: Gate 2 Post-Treatment Output & Physical Sanity Gate (`src/fireedge/validation/output_validator.py`)
* **Objective:** Run *after* the CNN has made its prediction to eliminate false alarms and hallucinations before queuing data for downlink.
* **Key Sanity Checks:**
  - **Thermal Elevation Check ($\Delta T \ge 10\text{ K}$):** A fire pixel predicted by the CNN **must** be physically hotter than the surrounding background by at least $10\text{ K}$. If the CNN hallucinates a fire on a cold cloud edge, snow, or coast, Gate 2 suppresses the false alarm.
  - **Spatial Plausibility:** Discards isolated single-pixel speckles and rejects full-frame collapse (>50% fire hallucinations).
* **Current Status:** Implemented in `src/fireedge/validation/output_validator.py` with 7 passing unit tests (`tests/test_output_validator.py`).

#### Task 3.2: Compact CCSDS Alert Packet Generator (`src/fireedge/telemetry/alert_packet.py`)
* **Objective (§6.2):** Transmit only actionable intelligence (~200 bytes) rather than the raw multi-megabyte image.
* **Packet Schema:**
  - Header: Satellite ID, Timestamp (UTC), Orbit Counter (16 bytes)
  - Geolocation: Fire Centroid Latitude & Longitude (float32, 8 bytes)
  - Fire Radiative Power (FRP): Estimated radiant heat output (float32, 4 bytes)
  - Detection Metrics: Peak confidence score, fire pixel count, bounding box `[xmin, ymin, xmax, ymax]` (16 bytes)
  - Integrity: CRC32 checksum (4 bytes)
  - Total Size: **$\approx 200\text{ bytes}$** (a **$>99.9\%$ compression ratio** against raw frames).

#### Task 3.3: Priority Queue Downlink Scheduler & Edge Benchmarking (`src/fireedge/telemetry/queue.py`)
* **Objective (§6.3 John McDonald rule):** Handle communication loss and intermittent ground contact windows.
* **Key Deliverables:**
  - **Retriable Priority Queue:** Persists alerts in local non-volatile storage, ranked by fire intensity and confidence. If a ground station pass is interrupted, unsent alerts remain queued for the next pass.
  - **Hardware Benchmark Suite (§6.6):** Measures execution latency (FPS), peak memory usage (RAM), and estimated energy per frame (Joules/frame) on target hardware.

### 5.2 How Person 3 Works Without Waiting for Person 2
* Person 3 already has `output_validator.py` and can use a simple threshold-based dummy segmenter (`prob = (thermal_img > threshold).astype(float)`) to build and verify Gate 2 and the alert packet generator immediately.

---

## 6. Summary Comparison Matrix: Roles & Ownership

| Category | Person 1 (You) ✅ | Person 2 | Person 3 |
| :--- | :--- | :--- | :--- |
| **Pipeline Stage** | **Stages 1 & 2: In-Flight Validation, Preprocessing & Data** | **Stage 3: Edge CNN Model** | **Gate 2 Output Sanity & Stage 4: Alert Telemetry** |
| **Status** | **✅ COMPLETE** | 🔲 Ready to Start | 🔲 Gate 2 Done, Telemetry & Queue Remaining |
| **Core Skillset** | Remote Sensing Radiometry, In-Flight Sensor Quality Gates (Gate 1), PyTorch DataLoaders | Deep Learning, CNN Architectures, Quantization | Aerospace Systems, Output QA, Embedded CCSDS Telemetry |
| **Input** | Raw detector frames (in-flight & ground) & GeoTIFFs | Normalized PyTorch batches `[B, 1, 256, 256]` | CNN fire probability maps & thermal frames (Gate 2) |
| **Output Deliverable** | In-flight Gate 1 telemetry validator (`FlightGateValidator`), calibrated tensors & DataLoaders | Trained weights `model.onnx` & `infer()` function | Validated CCSDS alert packet (~200 bytes) & Downlink Queue |
| **Primary Code Directory** | `src/fireedge/validation/` (Gate 1)<br>`src/fireedge/preprocessing/`<br>`src/fireedge/data/`<br>`src/fireedge/io/`<br>`src/fireedge/exploration/` | `src/fireedge/models/`<br>`src/fireedge/training/`<br>`src/fireedge/quantization/` | `src/fireedge/validation/output_validator.py`<br>`src/fireedge/telemetry/`<br>`src/fireedge/benchmarking/` |
| **Primary Test Suite** | `tests/test_flight_validator.py`<br>`tests/test_frame_validator.py`<br>`tests/test_preprocessing.py`<br>`tests/test_data_dataset.py`<br>`tests/test_io_*.py` | `tests/test_models.py`<br>`tests/test_losses.py`<br>`tests/test_quantization.py` | `tests/test_output_validator.py`<br>`tests/test_telemetry.py` |
| **Tests Passing** | **34 / 34** | — | 7 / 7 (Gate 2) |

---

## 7. Person 1 Completion Checklist

| # | Task | Module | Status |
| :--- | :--- | :--- | :--- |
| 1.0 | Gate 1 In-Flight Sensor Validation (PSF Spatial Halo, SEU Rejection) | `src/fireedge/validation/flight_gate_validator.py` | ✅ Done |
| 1.1 | Gate 1 Ground Mode Validation (Dual-Band Cross-Confirmation) | `src/fireedge/validation/frame_validator.py` | ✅ Done |
| 1.2 | Radiometric Planck Calibration (DN $\to$ Kelvin) | `src/fireedge/preprocessing/calibration.py` | ✅ Done |
| 1.3 | Microbolometer NUC & Degrader Simulator | `src/fireedge/preprocessing/nuc.py` | ✅ Done |
| 1.4 | Dynamic Thermal Normalization [0.0, 1.0] | `src/fireedge/preprocessing/normalization.py` | ✅ Done |
| 1.5 | PyTorch ActiveFireDataset | `src/fireedge/data/dataset.py` | ✅ Done |
| 1.6 | Pan-African DataLoader & Hard-Negative Mining | `src/fireedge/data/datamodule.py` | ✅ Done |
| 1.7 | ActiveFire I/O Loader & Region Classifier | `src/fireedge/io/activefire.py` | ✅ Done |
| 1.8 | TS-SatFire I/O Loader (Decimated VIIRS) | `src/fireedge/io/tssatfire.py` | ✅ Done |
| 1.9 | THRawS I/O Loader (Level-0 Raw Bridge) | `src/fireedge/io/thraws.py` | ✅ Done |
| 1.10 | EDA Exploration Modules | `src/fireedge/exploration/` | ✅ Done |
| 1.11 | Interactive EDA Notebook (Verified, Zero Errors) | `notebooks/01_african_data_exploration.ipynb` | ✅ Done |
| 1.12 | Gate 1 Dataset Curation & Reporting Scripts | `scripts/20_validate_dataset.py`, `scripts/21_validation_report.py` | ✅ Done |
| 1.13 | End-to-End Pipeline Verification Script | `scripts/verify_pipeline.py` | ✅ Done |
| 1.14 | Unit Tests (34 passing) | `tests/` | ✅ Done |
