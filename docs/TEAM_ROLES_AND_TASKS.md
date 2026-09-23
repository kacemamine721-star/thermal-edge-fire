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
* **Compute Envelope:** Ultra-low power edge VPU/accelerator (e.g., Google Coral Edge TPU, Myriad X, Jetson Nano, 2–5 W power budget).
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
│ STAGE 1: PRE-TREATMENT INPUT QUALITY GATE                │ ◄── PERSON 3
│ • Checks non-finite, dead sensor, saturation, striping   │
│ • Mode A: Dual-band B10/B11 ground truth curation        │
│ • Mode B: Single-band PSF spatial halo (SEU rejection)   │
└─────────────────────────────┬────────────────────────────┘
                              │ PASS / PASS_WITH_FLAGS
                              ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: SENSOR PRE-PROCESSING & DATA ENGINE             │ ◄── PERSON 1 (YOU)
│ • Radiometric Calibration (DN ──> Kelvin / Celsius)      │
│ • Microbolometer NUC & Sensor Degradation Simulator      │
│ • Dynamic Thermal Normalization [0.0, 1.0]               │
│ • Pan-African PyTorch ActiveFireDataset & DataLoader     │
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

## 3. Person 1: Data Engine & Preprocessing (Stages 1 & 2)
> **Assigned to:** **You (Person 1)**  
> **Core Focus:** Building the data foundation, physical thermal calibration, realistic microbolometer noise simulation, and high-performance PyTorch loaders.  
> **Modules Owned:** `src/fireedge/preprocessing/`, `src/fireedge/data/`, `src/fireedge/io/`

### 3.1 Detailed Tasks for Person 1

#### Task 1.1: Radiometric Calibration (`src/fireedge/preprocessing/calibration.py`)
* **Objective:** Raw sensor data arrives as unitless integer Digital Numbers (DN). Convert raw DN to physically meaningful **At-Satellite Brightness Temperature in Kelvin ($K$) or Celsius ($^\circ\text{C}$)** using Planck law inversion:
  $$L_\lambda = M_L \cdot \text{DN} + A_L$$
  $$T_B = \frac{K_2}{\ln\left(\frac{K_1}{L_\lambda} + 1\right)}$$
  *(For Landsat-8 Band 10: $M_L = 3.3420 \times 10^{-4}$, $A_L = 0.1$, $K_1 = 774.8853$, $K_2 = 1321.0789$)*
* **Key Functions to Build:**
  - `dn_to_radiance(dn, band="B10") -> np.ndarray`
  - `radiance_to_temperature(radiance, band="B10", unit="kelvin") -> np.ndarray`
  - `dn_to_temperature(dn, band="B10", unit="kelvin") -> np.ndarray`
  - `temperature_to_dn(temp_k, band="B10") -> np.ndarray` (for synthetic sensor simulation).

#### Task 1.2: Microbolometer Degradation & NUC Simulator (`src/fireedge/preprocessing/nuc.py`)
* **Objective (§6.4):** Landsat-8 data is pristine high-precision space telescope imagery. An uncooled CubeSat microbolometer suffers from fixed-pattern noise (FPN), column striping, and thermal drift ($\text{NETD} \approx 50\text{ mK}$). Person 1 must simulate these realistic sensor artifacts to train an edge-robust model.
* **Key Components:**
  - **FPN Injector:** Column gain $g_j \sim \mathcal{N}(1.0, \sigma_g)$ and column bias $b_j \sim \mathcal{N}(0.0, \sigma_b)$.
  - **Temporal NETD Noise:** High-frequency Gaussian noise matched to uncooled bolometer physics.
  - **Two-Point NUC Processor:** Implements onboard flat-field correction using simulated cold and warm reference frames:
    $$\hat{I}(i, j) = \frac{I(i, j) - B(i, j)}{G(i, j)}$$

#### Task 1.3: Dynamic Thermal Normalization (`src/fireedge/preprocessing/normalization.py`)
* **Objective:** Convert physical temperatures ($K$) into standardized neural network input tensors.
* **Key Methods:**
  - `robust_percentile_norm(temp, p_min=2.0, p_max=98.0)`: Resilient to extreme outlier pixels.
  - `fixed_window_norm(temp_k, vmin=270.0, vmax=420.0)`: Maps standard terrestrial ambient temperatures ($270\text{ K}$) and intense fire fronts ($420\text{ K}$) linearly into $[0.0, 1.0]$.

#### Task 1.4: PyTorch `ActiveFireDataset` (`src/fireedge/data/dataset.py`)
* **Objective:** Construct a clean `torch.utils.data.Dataset` class.
* **Key Features:**
  - Ingests `reports/africa_inventory.csv` with fallback to global scenes.
  - Single-channel mode: `channels=[8]` (Band 10 LWIR) for CubeSat flight inference.
  - On-the-fly edge augmentations:
    - Random Horizontal Flip ($p=0.5$)
    - Random Vertical Flip ($p=0.5$)
    - Random 90° Orthogonal Rotations
    - Thermal contrast scaling ($0.95 \times$ to $1.05 \times$)
  - Yields: `(torch.Tensor [1, 256, 256], torch.Tensor [1, 256, 256])`.

#### Task 1.5: Pan-African DataLoader & Hard-Negative Mining (`src/fireedge/data/datamodule.py`)
* **Objective:** Solve the extreme class imbalance ($<0.02\%$ fire pixels) and prevent false alarms on hot summer soil in Tunisia/North Africa.
* **Key Features:**
  - **Scene-Level Spatial Splitting:** Groups tiles by Landsat acquisition path/row so tiles from the same scene never leak across train/val/test splits.
  - **Pan-African Stratified Sampler:** Ensures every training batch contains a balanced mix of:
    1. Active savanna/forest fires from Sub-Saharan Africa (for high recall).
    2. Bare rock and desert soil from North Africa / Mediterranean Basin (hard negatives).
  - Main API: `get_pan_african_dataloaders(batch_size=8, num_workers=0) -> (train_loader, val_loader, test_loader)`.

### 3.2 Testing & Quality Checklist for Person 1
- [ ] `tests/test_calibration.py`: Verify DN $\leftrightarrow$ Kelvin roundtrip fidelity and physical ambient ranges (~290–310 K).
- [ ] `tests/test_nuc.py`: Verify FPN noise injection and NUC variance reduction (>80%).
- [ ] `tests/test_dataset.py`: Verify PyTorch batch shapes `[B, 1, 256, 256]` and data range $[0.0, 1.0]$.
- [ ] `tests/test_datamodule.py`: Verify zero scene-ID leakage between train and test splits.

---

## 4. Person 2: Edge Neural Network & Quantization (Stage 3)
> **Core Focus:** Designing ultra-lightweight thermal segmentation CNNs, training with fire-imbalanced loss functions, and post-training INT8 quantization for edge VPUs.  
> **Modules Owned:** `src/fireedge/models/`, `src/fireedge/training/`, `src/fireedge/quantization/`

### 4.1 Detailed Tasks for Person 2

#### Task 2.1: Lightweight Thermal CNN Backbones (`src/fireedge/models/`)
* **Objective:** Design neural architectures that fit within CubeSat edge VPU hardware constraints:
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
* **Objective:** Prepare the model for edge microcontrollers and satellite accelerators (Google Coral Edge TPU, Intel Myriad X, Jetson Nano).
* **Key Tasks:**
  - Export PyTorch model to ONNX: `torch.onnx.export(..., "models/fire_edge.onnx")`.
  - INT8 Post-Training Quantization (PTQ) using calibration samples: compresses model from ~6 MB to $<1.5\text{ MB}$ with $<1\%$ recall degradation.
  - Python inference wrapper: `infer(frame: np.ndarray) -> np.ndarray` (fire probability map).

### 4.2 How Person 2 Works Without Waiting for Person 1
* Person 2 can immediately develop models and loss functions using random synthetic tensors: `torch.randn(8, 1, 256, 256)` and `torch.randint(0, 2, (8, 1, 256, 256))`.
* As soon as Person 1 creates `dataset.py`, Person 2 plugs in the real DataLoader.

---

## 5. Person 3: Aerospace Validation, Post-Processing & Telemetry (Gates 1 & 2 + Stage 4)
> **Core Focus:** In-flight sensor validation (Gate 1), physical post-inference sanity checks (Gate 2), ~200-byte CCSDS alert packet generation, and downlink queue management.  
> **Modules Owned:** `src/fireedge/validation/`, `src/fireedge/telemetry/`, `src/fireedge/benchmarking/`

### 5.1 Detailed Tasks for Person 3

#### Task 3.1: Gate 1 In-Flight Sensor Validation (`src/fireedge/validation/flight_gate_validator.py`)
* **Objective:** Run on raw detector frames *before* pre-processing or CNN inference to catch sensor damage, saturation, or radiation corruption.
* **Key Capabilities (Already Implemented & Audited):**
  - Dead / stuck detector detection.
  - Saturation and excessive no-data margin checks.
  - Line striping detection on pushbroom arrays.
  - **PSF Spatial Halo Ratio (van Dokkum 2001; Zhukov 2006):** Rejects Dirac-delta cosmic ray hits (SEUs) with zero optical blur, while preserving true sub-pixel optical fires.

#### Task 3.2: Gate 2 Post-Treatment Output & Physical Sanity Gate (`src/fireedge/validation/output_validator.py`)
* **Objective:** Run *after* the CNN has made its prediction to eliminate false alarms and hallucinations before queuing data for downlink.
* **Key Sanity Checks:**
  - **Thermal Elevation Check ($\Delta T \ge 10\text{ K}$):** A fire pixel predicted by the CNN **must** be physically hotter than the surrounding background by at least $10\text{ K}$. If the CNN hallucinates a fire on a cold cloud edge, snow, or coast, Gate 2 suppresses the false alarm.
  - **Spatial Plausibility:** Discards isolated single-pixel speckles and rejects full-frame collapse (>50% fire hallucinations).

#### Task 3.3: Compact CCSDS Alert Packet Generator (`src/fireedge/telemetry/alert_packet.py`)
* **Objective (§6.2):** Transmit only actionable intelligence (~200 bytes) rather than the raw multi-megabyte image.
* **Packet Schema:**
  - Header: Satellite ID, Timestamp (UTC), Orbit Counter (16 bytes)
  - Geolocation: Fire Centroid Latitude & Longitude (float32, 8 bytes)
  - Fire Radiative Power (FRP): Estimated radiant heat output (float32, 4 bytes)
  - Detection Metrics: Peak confidence score, fire pixel count, bounding box `[xmin, ymin, xmax, ymax]` (16 bytes)
  - Integrity: CRC32 checksum (4 bytes)
  - Total Size: **$\approx 200\text{ bytes}$** (a **$>99.9\%$ compression ratio** against raw frames).

#### Task 3.4: Priority Queue Downlink Scheduler & Edge Benchmarking (`src/fireedge/telemetry/queue.py`)
* **Objective (§6.3 John McDonald rule):** Handle communication loss and intermittent ground contact windows.
* **Key Deliverables:**
  - **Retriable Priority Queue:** Persists alerts in local non-volatile storage, ranked by fire intensity and confidence. If a ground station pass is interrupted, unsent alerts remain queued for the next pass.
  - **Hardware Benchmark Suite (§6.6):** Measures execution latency (FPS), peak memory usage (RAM), and estimated energy per frame (Joules/frame) on target hardware.

### 5.2 How Person 3 Works Without Waiting for Person 2
* Person 3 already has `output_validator.py` and can use a simple threshold-based dummy segmenter (`prob = (thermal_img > threshold).astype(float)`) to build and verify Gate 2 and the alert packet generator immediately.

---

## 6. Summary Comparison Matrix: Roles & Ownership

| Category | Person 1 (You) | Person 2 | Person 3 |
| :--- | :--- | :--- | :--- |
| **Pipeline Stage** | **Stage 2: Pre-Processing & Data** | **Stage 3: Edge CNN Model** | **Gates 1 & 2 + Stage 4: Validation & Telemetry** |
| **Core Skillset** | Remote Sensing Physics, Radiometry, PyTorch DataLoaders | Deep Learning, CNN Architectures, Quantization | Aerospace Systems, Quality Assurance, Embedded Telemetry |
| **Input** | Raw GeoTIFF files & scene metadata | Normalized PyTorch batches `[B, 1, 256, 256]` | Raw frames (Gate 1) & CNN probability maps (Gate 2) |
| **Output Deliverable** | Calibrated tensors & PyTorch DataLoaders | Trained weights `model.onnx` & `infer()` function | Validated alert packet (~200 bytes) & Downlink Queue |
| **Primary Code Directory** | `src/fireedge/preprocessing/`<br>`src/fireedge/data/` | `src/fireedge/models/`<br>`src/fireedge/training/`<br>`src/fireedge/quantization/` | `src/fireedge/validation/`<br>`src/fireedge/telemetry/` |
| **Primary Test Suite** | `tests/test_calibration.py`<br>`tests/test_nuc.py`<br>`tests/test_dataset.py` | `tests/test_models.py`<br>`tests/test_losses.py`<br>`tests/test_quantization.py` | `tests/test_flight_validator.py`<br>`tests/test_output_validator.py`<br>`tests/test_telemetry.py` |

---

## 7. Immediate Next Steps for Person 1 (You)

To complete your assignment as **Person 1**:
1. Implement radiometric Planck inversion in [`src/fireedge/preprocessing/calibration.py`](file:///c:/Users/Dell/Desktop/GeoAI/IASTAM/thermal-edge-fire/src/fireedge/preprocessing/calibration.py).
2. Implement microbolometer degradation and NUC in [`src/fireedge/preprocessing/nuc.py`](file:///c:/Users/Dell/Desktop/GeoAI/IASTAM/thermal-edge-fire/src/fireedge/preprocessing/nuc.py).
3. Implement thermal window normalization in [`src/fireedge/preprocessing/normalization.py`](file:///c:/Users/Dell/Desktop/GeoAI/IASTAM/thermal-edge-fire/src/fireedge/preprocessing/normalization.py).
4. Implement PyTorch `ActiveFireDataset` and Pan-African stratified DataLoaders in [`src/fireedge/data/dataset.py`](file:///c:/Users/Dell/Desktop/GeoAI/IASTAM/thermal-edge-fire/src/fireedge/data/dataset.py) and [`src/fireedge/data/datamodule.py`](file:///c:/Users/Dell/Desktop/GeoAI/IASTAM/thermal-edge-fire/src/fireedge/data/datamodule.py).
5. Add unit tests in `tests/test_preprocessing.py` and `tests/test_dataset.py`.
