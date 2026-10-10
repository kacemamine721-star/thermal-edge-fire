# IASTAM / Thermal-Edge-Fire
## Model Architecture, Evaluation Results, and Interactive Demo Technical Documentation

> **Document status:** Corrected technical documentation for the repository implementation and checked-in benchmark outputs  
> **Project:** IASTAM / SpaceTOON — Thermal-Edge-Fire  
> **Intended deployment concept:** Edge processing for a 6U-class Earth-observation mission  
> **Primary evaluated configuration:** Pereira-3C (`C7 + C6 + C2`: SWIR2 + SWIR1 + Blue)  
> **Model family:** Tiny U-Net, ONNX Runtime FP32 and INT8  
> **Documentation date:** October 2026  
> **Repository:** <https://github.com/kacemamine721-star/thermal-edge-fire>

---

## 1. Scope and interpretation

Thermal-Edge-Fire is a **software prototype and evaluated information-extraction pipeline** intended to inform an onboard wildfire-detection architecture for a 6U-class CubeSat. It combines multispectral segmentation, input-quality screening, connected-component event extraction, binary alert packets, and software studies of downlink prioritization.

The results in this document describe the configurations and experiments available in this repository. They do **not** establish that the pipeline is flight-qualified, that it has been run on a CubeSat, or that it has been benchmarked on a Raspberry Pi Compute Module 4 or another flight computer.

The primary evaluated model, **Pereira-3C**, receives Landsat-8-style reflective bands C7/C6/C2. It is not an LWIR-microbolometer model. The B10-only configuration is a thermal-band control experiment evaluated against the same Pereira voting masks; because those masks are generated from reflectance/SWIR algorithms, its score measures performance against that particular reference-label source. It does not validate or rule out a different, properly calibrated thermal payload.

| Item | Status supported by this repository |
|---|---|
| Six-stage edge information-extraction architecture | Implemented software prototype |
| Pereira-3C Tiny U-Net | Trained and evaluated against the Pereira voting-mask benchmark |
| B10-only thermal input | Evaluated control configuration; not an LWIR flight-payload validation |
| Pereira-3C + B10 hybrid | Evaluated comparison configuration |
| ONNX FP32 and INT8 export | Bundled and executable in the demo on a compatible host |
| Raspberry Pi CM4 / flight-computer performance | Not measured in this repository |
| Flight sensor, spacecraft power, radiation qualification, orbital operations, RF link | Not validated end to end |

The central contribution is the **information-extraction chain**—turning an image tile into structured event information—rather than a claim that a specific spacecraft sensor has already been validated in orbit.

## 2. Executive summary

### 2.1 Headline results

| Metric | Checked-in result | What the result means |
|---|---:|---|
| Pereira-3C parameters | 482,737 | Parameter count of the Tiny U-Net model |
| Pereira-3C computation | 2.31945 GMAC per 256×256 tile | Model-compute estimate; not a power measurement |
| INT8 ONNX file | 527,200 bytes (about 514.8 KiB) | Model-file size only; not total runtime-memory use |
| FP32 ONNX file | 1,937,578 bytes (about 1.85 MiB) | Bundled FP32 model-file size |
| Pereira-3C test F1 / IoU | 0.94925 / 0.90340 | Pixel-mask metrics at the selected threshold on the benchmark test split |
| Pereira-3C frame recall / frame FAR | 0.96477 / 0.06347 | 96.48% frame recall and 6.35% frame false-alarm rate on that split |
| Pereira-3C event recall / event precision | 0.95525 / 0.98362 | Tolerance-based event metrics against the benchmark reference |
| Manual-label zero-shot F1 / recall | 0.84935 / 0.95899 | Reported independent manual-label evaluation without fine-tuning; not a universal operating guarantee |
| Host pipeline latency | Mean 65.68 ms; p50 64.38 ms; p95 77.68 ms; p99 91.30 ms | Measured on an Intel Xeon host; not an onboard/flight-computer result |
| One-event packet | 48 bytes | Protocol accounting result for one encoded event |
| Maximum six-event packet | 183 bytes | Protocol accounting result for six encoded events |
| Fire-free heartbeat | 21 bytes | Optional pipeline heartbeat; the web demo disables it by default |
| Aggregate data reduction | 6,234.99× over 1,500 simulated frames including heartbeats | Software/accounting simulation, not a measured RF-link result |

### 2.2 Primary result in context

At their selected operating points, Pereira-3C has the highest test F1 and IoU of the three reported configurations. The hybrid configuration has higher pixel precision and slightly higher F1 at the separate `F1@0.25` comparison threshold; therefore Pereira-3C should not be described as best at every threshold or metric.

The test split contains 3,946 voting-fire patches and 835 voting-zero patches (4,781 in total). About 82.5% of test patches are positive, so the split is fire-heavy compared with many operational monitoring settings. Deployment false-alert workload must be evaluated under realistic prevalence and acquisition conditions as well as with the reported benchmark metrics.

## 3. Six-stage processing architecture

The software pipeline is organized into six stages. Gate 1 and Gate 2 are software safeguards around the segmentation model; they are not proof of radiation tolerance or physical qualification.

```text
┌───────────────────────────────────────────────────────────────────────────┐
│ 1. Gate 1 — Input-quality screening                                        │
│    Validate tile input, assess invalid/saturated data and dead columns;     │
│    optionally scrub isolated saturated pixels in the first input band.     │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 2. Fixed input normalization                                              │
│    Apply the training/deployment preprocessing contract to the selected    │
│    16-bit DN bands. Metadata-based radiometric validation remains required.│
└────────────────────────────────────┬──────────────────────────────────────┘
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 3. Neural segmentation                                                    │
│    Run the Tiny U-Net through ONNX Runtime (bundled FP32 or INT8 model).   │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 4. Probability transformation and thresholding                            │
│    Scale logits by temperature T, apply sigmoid, then threshold the map.  │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 5. Event extraction                                                      │
│    Label 8-connected components; compute bounding box, centroid, area,     │
│    mean temperature-scaled pixel probability, and SWIR2 contrast.          │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 6. Gate 2 and packet generation                                           │
│    Check output values/geometry, raise a warning for excessive mask area,  │
│    serialize up to six event records, and append a CRC32 checksum.         │
└───────────────────────────────────────────────────────────────────────────┘
```

The surrounding repository also contains a software model of a persistent, priority-based downlink queue. The queue study is separate from the six per-tile inference stages and does not simulate a validated flight radio or orbital contact schedule.

### Stage 1 — Gate 1 input-quality screening

Implemented in `demo/pipeline.py` as `Gate1` and used by the interactive app. Its configured checks include the valid-pixel fraction, saturation, and columns with very low variation. It identifies isolated saturated pixels in the first selected channel (C7/SWIR2). If `gate1_scrub` is enabled, the current implementation applies a 3×3 median filter to those isolated pixels in C7 and sets a status flag.

The robustness experiments inject synthetic noise, transient spikes, and dead columns. These are software stress tests, not physical radiation testing or a demonstration of tolerance to all sensor faults.

### Stage 2 — Fixed input normalization

The model path applies the following fixed transformation to the selected inputs:

\[
  x = \operatorname{clip}(DN \times 2.0\times10^{-5} - 0.1,\,0,\,1.6)
\]

The output is float32 in `[0, 1.6]`. In the current demo path the function is named `preprocess_dn` and its code/docstring calls the result TOA reflectance. For technical precision, the equation should be described as the **repository's fixed DN-normalization contract** unless the exact source-product metadata and preprocessing lineage have been verified. Landsat Level-1 TOA reflectance conversion uses band-specific rescaling coefficients in the scene metadata, plus a solar-angle correction for sun-corrected reflectance. See the references in Section 15.

### Stage 3 — Neural inference

`InferenceEngine` creates an ONNX Runtime CPU session and runs a warm-up inference. The model input is a single float32 tensor shaped `[1, 3, 256, 256]`; logits are returned for a 256×256 mask. The interactive app can select the bundled Pereira-3C INT8 or FP32 model. The current bundle contains these two Pereira-3C ONNX files; it does **not** contain separate B10-only or four-channel hybrid ONNX models.

### Stage 4 — Temperature scaling and decision threshold

The code applies scalar logit temperature scaling followed by the sigmoid function:

\[
  p = \sigma(z/T), \qquad \sigma(u)=\frac{1}{1+e^{-u}}
\]

For the primary Pereira-3C benchmark, `T = 2.1227591254` and the selected threshold is `τ = 0.96`. The threshold was selected on the validation data using the `constrained_plateau_f1` rule, with a target frame recall of at least 0.95 and frame FAR of at most 0.10. The manifest records that these constraints were met on validation; this is not a guarantee that they will be met on future imagery.

The B10 control and hybrid model use their own selected temperatures and thresholds (Section 6). Do not apply the Pereira-3C threshold to all configurations by assumption.

### Stage 5 — Connected components and event features

The current implementation uses 8-connectivity for connected-component labeling. For each component it computes a pixel-space bounding box, pixel centroid, pixel area, approximate area in hectares based on the supplied GSD (default 30 m), and an SWIR2 contrast measure relative to the median C7 value in the tile. Geolocation is computed from the GeoTIFF affine transform and CRS when valid georeferencing is available; otherwise the event is flagged as having no valid geolocation.

`MIN_EVENT_PIXELS = 1`: single-pixel detections are retained by the current demo. The documentation therefore does not claim that the pipeline automatically filters single-pixel clusters.

**Confidence semantics:** `demo/pipeline.py` currently sets event confidence to the mean of the temperature-scaled pixel probabilities within the detected component. Although the notebook/reporting workflow includes event-level isotonic calibration, the checked-in demo pipeline does not load and apply that event-calibration table. In this demo, confidence must therefore not be described as an isotonic-calibrated probability that an event is a real wildfire.

### Stage 6 — Gate 2 and packet encoding

`Gate2` verifies that the probability map is finite and within the accepted numeric range and checks basic event bounding-box/area bounds. When the predicted mask covers more than 5% of the tile, it returns a **warning** rather than automatically rejecting the output. Passing Gate 2 is a software consistency check, not a proof that a detection is physically a wildfire.

The packet encoder writes a little-endian version-3 binary packet with a CRC32 checksum. Protocol details are in Section 10.

## 4. Sensor bands and dataset foundations

### 4.1 Compared configurations

The repository manifest specifies three configurations:

| Configuration | Input channels | Purpose in the experiment |
|---|---|---|
| `pereira3c` | C7 + C6 + C2 | Primary evaluated reflective-band configuration: SWIR2, SWIR1, Blue |
| `b10` | B10 | Thermal-band control experiment |
| `pereira3c_b10` | C7 + C6 + C2 + B10 | Hybrid comparison |

Here C7, C6 and C2 refer to Landsat-8 Band 7 (SWIR2), Band 6 (SWIR1) and Band 2 (Blue), respectively. Landsat-8 Band 10 is a thermal-infrared band with a native ground resolution of about 100 m, commonly distributed resampled to a 30 m grid. Resampling the band to a 30 m grid does not provide native 30 m thermal resolution. See the USGS band-designations reference in Section 15.

### 4.2 Physical motivation and limits

Wien's displacement law for an ideal blackbody is

\[
  \lambda_{\max} \approx \frac{2898}{T}\;\mu\mathrm{m}
\]

which places the idealized radiance peak of a 300 K body near 9.7 μm and that of 800–1200 K material near about 3.6–2.4 μm. This helps explain why SWIR can be informative for hot combustion. It is not, by itself, proof that SWIR always outperforms thermal sensing in every instrument, scene, or fire condition. Real satellite measurements are influenced by sub-pixel mixing, atmospheric effects, reflected sunlight, instrument spectral response, saturation, surface properties, and spatial resolution.

The repository's band-information probe, as summarized in the README, reports the following pixel-level AUC values against the benchmark's Pereira voting masks:

| Band | Approximate spectral role | Reported pixel-level AUC |
|---|---|---:|
| C7 / SWIR2 | Strong input signal for the current reflectance-derived reference | 0.99276 (about 0.993) |
| B10 / thermal infrared | Thermal control input | 0.86520 (about 0.865) |
| C6 / SWIR1 | Supporting spectral context | 0.82013 (about 0.820) |
| C2 / Blue | Supporting visible-band context | 0.56489 (about 0.565) |

These AUC results support choosing C7/C6/C2 for this **specific benchmark**. They do not establish that reflective-band algorithms are the correct ground truth for thermal sensing, and they do not independently prove a universal sensor-physics ranking.

### 4.3 Dataset splits

The split is described in `demo/models/manifest_v7.json` as scene-level 70/15/15, seed 42. The checked-in split summary is:

| Split | Voting-fire tiles | Voting-zero tiles | Total tiles | Positive proportion |
|---|---:|---:|---:|---:|
| Train | 15,377 | 3,554 | 18,931 | 81.24% |
| Validation | 3,177 | 840 | 4,017 | 79.09% |
| Test | 3,946 | 835 | 4,781 | 82.51% |
| **Total** | **22,500** | **5,229** | **27,729** | **81.14%** |

The labels are Pereira voting masks formed from agreement among reflectance-based fire algorithms; the manifest states that thermal B10 was not used to generate the voting reference. These labels are an algorithmic benchmark reference, not comprehensive field-confirmed wildfire truth.

The positive-patch proportion is high in all splits, including the test set. Frame FAR is reported on the fire-free reference patches, but a deployed system's false-alert workload depends on the real prevalence of fire-free acquisitions as well as on the per-frame FAR.

## 5. Model architecture and artifact sizes

The evaluated segmentation model is a Tiny U-Net with a base channel width of 16 and standard 3×3 convolutions. The documented feature widths progress through 16, 32 and 64 channels, with a 128-channel bottleneck. The primary model has 482,737 parameters and an estimated 2.319450112 GMAC per 256×256 three-channel tile.

| Artifact | Size | Notes |
|---|---:|---|
| `demo/models/unet_pereira3c_fp32.onnx` | 1,937,578 bytes (~1.85 MiB) | FP32 ONNX model |
| `demo/models/unet_pereira3c_int8.onnx` | 527,200 bytes (~514.8 KiB) | INT8 ONNX model |
| File-size reduction | ~3.68× | Model-file comparison only |

A model file's byte size alone does not demonstrate that the complete inference runtime, activations, workspace, libraries, and operating system fit in a specified SRAM or L2 cache. The repository does not include a complete onboard peak-memory or power measurement. The exported ONNX models are tested through the host demo; no target-board or flight-computer measurement is claimed.

## 6. Held-out test results across configurations

The following numbers are from `reports/model_benchmarks/test_results_by_config.csv`, at each configuration's selected operating point.

| Configuration | Bands | Threshold | Precision | Recall | IoU | F1 | Frame recall | Frame FAR | Event recall | Event precision | Validation constraints met? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`pereira3c`** | C7+C6+C2 | 0.96 | 0.94957 | 0.94892 | 0.90340 | **0.94925** | 0.96477 | 0.06347 | 0.95525 | 0.98362 | Yes |
| `b10` | B10 | 0.29 | 0.55338 | 0.32798 | 0.25933 | **0.41186** | 0.18018 | 0.02635 | 0.13263 | 0.83008 | No |
| `pereira3c_b10` | C7+C6+C2+B10 | 0.60 | 0.96627 | 0.92634 | 0.89733 | **0.94589** | 0.96427 | 0.06707 | 0.94545 | 0.98402 | Yes |

Metrics are shown to five decimal places where useful; the CSV is the authoritative source for the exact values.

### Interpretation

1. **Pereira-3C is the primary result at the selected operating point.** It has the highest F1 (0.94925) and IoU (0.90340) among these three selected points. Its test frame recall is about 96.48%, and its frame FAR is about 6.35%.
2. **B10-only has much lower recall against the current voting-mask benchmark.** Its frame FAR is lower (about 2.63%), but frame recall is only about 18.02% and event recall about 13.26%. The manifest records that the selection constraints were infeasible. Because the voting masks were generated from reflectance/SWIR algorithms, the B10 score measures the B10-to-reference-label gap in this experiment; it is not a validation or rejection of a future thermal-native pipeline.
3. **The hybrid configuration increases precision but lowers recall compared with Pereira-3C at the selected points.** Precision is 0.96627, recall 0.92634, F1 0.94589, IoU 0.89733, frame FAR 0.06707, event recall 0.94545, and event precision 0.98402.
4. The separate `F1@0.25` column in the CSV is 0.94864 for Pereira-3C and 0.94931 for the hybrid configuration. Thus, Pereira-3C should be described as the best primary operating point by its selected F1/IoU—not as the best model under every threshold or metric.

## 7. Temperature calibration and operating-point selection

The primary configuration uses the scalar temperature `T = 2.122759125440246`, threshold `τ = 0.96`, and selection rule `constrained_plateau_f1`. The stated validation targets are frame recall ≥ 0.95 and frame FAR ≤ 0.10. The manifest reports `constraints_met_on_validation = true` for Pereira-3C and the hybrid configuration, and `false` for B10.

| Configuration | Temperature | Selected threshold | Selection rule | Validation constraints met |
|---|---:|---:|---|---|
| `pereira3c` | 2.122759 | 0.96 | `constrained_plateau_f1` | Yes |
| `b10` | 1.969283 | 0.29 | `infeasible_fallback_far_limited_plateau_f1` | No |
| `pereira3c_b10` | 2.177020 | 0.60 | `constrained_plateau_f1` | Yes |

Temperature scaling changes the logits before sigmoid. It should not be called Platt scaling in this document; the implementation uses scalar logit temperature scaling. Validation selection results do not constitute guarantees under sensor shifts, season/biome shifts, new label policies, or different fire prevalence.

## 8. Event-size sensitivity and event-matching definition

The final test-results CSV reports event recall by the size of the **reference connected components**:

| Reference component size | Pereira-3C event recall |
|---|---:|
| 1–5 pixels | 0.94564 |
| 6–25 pixels | 1.00000 |
| 26–100 pixels | 1.00000 |
| More than 100 pixels | 1.00000 |
| All reference events | 0.95525 |

These values are tolerance-based benchmark results, not an operational guarantee that every fire of a particular physical size will always be detected. The notebook's event-matching routine uses a three-pixel spatial tolerance around predicted/reference components; this should be disclosed when comparing event metrics. The area conversion in the demo also depends on the GSD provided by the GeoTIFF (with 30 m as a fallback), and a nominal 30 m pixel corresponds to 0.09 ha geometrically—not to a claim that every fire fills a pixel.

Do not convert these component-recall results into “100% deterministic frame recall for every fire larger than six pixels.” They describe the checked-in test split and the specified event-matching method only.

## 9. Manual-label zero-shot evaluation

`reports/model_benchmarks/manual_zero_shot.csv` reports a separate evaluation without fine-tuning on human-annotated active-fire scenes. The project notes describe 13 Landsat scenes (10 with fire and 3 fire-free) for this human-label check.

| Configuration | Precision | Recall | IoU | F1 | Frame recall | Frame FAR | Event recall | Event precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Pereira-3C** | 0.76220 | 0.95899 | 0.73814 | **0.84935** | 0.99 | 0.00 | 0.92897 | 0.80138 |
| B10 | 0.71537 | 0.27620 | 0.24886 | **0.39853** | 0.43 | 0.00 | 0.32759 | 0.92013 |
| Hybrid | 0.83863 | 0.94754 | 0.80142 | **0.88976** | 0.99 | 0.00 | 0.91103 | 0.81801 |

For Pereira-3C, the reported F1 is about 0.8493 and recall is about 0.9590. The reported 0.00 frame FAR applies to this finite manual evaluation only; it must not be presented as proof that the system has no false alarms in future operation. Manual-label testing is a useful circularity check because the main training/test voting reference is algorithm-generated, but it remains limited by the size and geographic composition of the manually annotated set.

## 10. Host latency and runtime measurements

### 10.1 Model-only ONNX benchmark

`reports/model_benchmarks/benchmark_host.csv` identifies the benchmark CPU as **Intel(R) Xeon(R) CPU @ 2.00GHz** and marks `is_onboard_measurement=False`.

| Model | Threads | Mean latency | p50 | p95 | Model-file size |
|---|---:|---:|---:|---:|---:|
| FP32 | 1 | 46.08 ms | 45.69 ms | 49.07 ms | 1.85 MiB |
| FP32 | 4 | 24.19 ms | 24.01 ms | 25.86 ms | 1.85 MiB |
| INT8 | 1 | 55.21 ms | 54.65 ms | 58.80 ms | 0.503 MiB |
| INT8 | 4 | 32.17 ms | 31.60 ms | 36.16 ms | 0.503 MiB |

The FP32 and INT8 results come from the reported host benchmark. The CSV field is named `file_MB`, but its values correspond to MiB (bytes divided by 1,048,576). INT8 has a smaller file, but it is not faster than FP32 in every host/thread setting. Quantization speed depends on the execution provider, kernel support, memory behavior, and hardware.

### 10.2 End-to-end software pipeline benchmark

`reports/model_benchmarks/pipeline_latency_host.csv` reports:

| Stage | Mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| Gate 1 | 1.772 ms | 1.740 ms | 3.577 ms | 4.505 ms |
| Preprocessing | 0.332 ms | 0.314 ms | 0.473 ms | 0.577 ms |
| Inference | 57.492 ms | 57.043 ms | 61.011 ms | 69.806 ms |
| Decision/event extraction | 5.926 ms | 4.152 ms | 15.397 ms | 27.584 ms |
| Gate 2 | 0.152 ms | 0.147 ms | 0.186 ms | 0.230 ms |
| Packet encoding | 0.045 ms | 0.045 ms | 0.083 ms | 0.103 ms |
| **Total pipeline** | **65.677 ms** | **64.376 ms** | **77.680 ms** | **91.300 ms** |

The arithmetic reciprocal of the mean total is approximately 15.2 tiles/s, but this is a host software estimate, not a spacecraft throughput guarantee. The often-used 100 ms comparison is a project-level reference target; it is not a verified orbital ground-track budget. Actual performance on an embedded processor must be measured with the intended runtime, memory limits, input I/O, and power configuration.

The demo can also run a 30-iteration benchmark on the machine hosting the app. In a constrained cloud deployment, its source code explicitly warns that those results are not representative of the reference x86 host or flight hardware.

## 11. Synthetic robustness evaluation

The checked-in `reports/model_benchmarks/robustness_primary.csv` contains the following results. “Noise,” “transients,” “dead columns,” and “quantization” refer to software-injected perturbations in the experiment; they are not physical radiation qualification levels.

| Scenario | Pixel F1 | Pixel recall | Frame recall | Frame FAR | Event recall (all) | False events / patch |
|---|---:|---:|---:|---:|---:|---:|
| Clean, scrub off | 0.95174 | 0.95392 | 0.96590 | 0.06704 | 0.95800 | 0.025 |
| Noise L1 | 0.95072 | 0.95290 | 0.96711 | 0.06145 | 0.95800 | 0.022 |
| Noise L2 | 0.94470 | 0.94959 | 0.96955 | 0.07821 | 0.95627 | 0.040 |
| Noise L3 | 0.84410 | 0.92253 | 0.96955 | 0.31285 | 0.94476 | 1.351 |
| Quantization L1 | 0.95188 | 0.95431 | 0.96590 | 0.06704 | 0.95857 | 0.026 |
| Quantization L2 | 0.95233 | 0.95495 | 0.96711 | 0.06704 | 0.95915 | 0.026 |
| Quantization L3 | 0.95257 | 0.95737 | 0.96833 | 0.06704 | 0.96203 | 0.024 |
| Transients L1, scrub off | 0.86460 | 0.95354 | 0.99635 | 0.82123 | 0.95742 | 1.601 |
| Transients L2, scrub off | 0.67180 | 0.95227 | 1.00000 | 1.00000 | 0.95282 | 6.526 |
| Transients L3, scrub off | 0.35121 | 0.94946 | 1.00000 | 1.00000 | 0.95167 | 26.577 |
| Dead columns L1 | 0.38844 | 0.95124 | 0.97564 | 0.36313 | 0.95685 | 1.954 |
| Dead columns L2 | 0.17993 | 0.94703 | 0.98904 | 0.63687 | 0.95627 | 6.263 |
| Dead columns L3 | 0.07493 | 0.93963 | 0.99391 | 0.92737 | 0.94707 | 16.863 |
| Transients L2 + Gate 1 scrub | 0.94311 | 0.93937 | 0.96468 | 0.07263 | 0.95685 | 0.051 |
| Transients L3 + Gate 1 scrub | 0.93979 | 0.93925 | 0.96346 | 0.07263 | 0.95627 | 0.103 |
| Clean + Gate 1 scrub* | 0.94468 | 0.93937 | 0.96468 | 0.06704 | 0.95685 | 0.025 |

\*The last row is labelled `clean` in the checked-in CSV. The notebook scenario configuration indicates it corresponds to the clean case with Gate 1 scrubbing enabled; this is a scenario-label inconsistency, not a separate clean/no-scrub result.

### Interpretation

Under the tested Level-2 transient injection, enabling Gate 1 scrubbing moves pixel F1 from 0.67180 to 0.94311 and frame FAR from 1.00000 to 0.07263. This is evidence that the tested scrubber can mitigate this particular synthetic perturbation. It does not justify the statement that radiation-induced false alarms have been eliminated in a physical space environment. The dead-column tests also show that some input defects remain highly damaging, so input QA should not be described as solving every fault mode.

## 12. Compact Binary Alert Packet Protocol

The packet format is version 3 and is encoded in little-endian order by `demo/pipeline.py`.

### 12.1 Packet layout

| Component | Size | Main contents |
|---|---:|---|
| Header | 17 bytes | Magic `0xF1`, version, model ID, frame sequence, timestamp, event count, status flags |
| Event record | 27 bytes each | Latitude/longitude integers, pixel bounding box, area in 0.1 ha units, confidence field, contrast metric, GSD, geolocation uncertainty, event flags |
| CRC32 trailer | 4 bytes | CRC32 over the header and event records |

The header does not contain a satellite ID in the current format; byte 2 is the **model ID**. The event record has fields for geolocation, image-space bounds, area, confidence, contrast, nominal GSD and uncertainty/status flags. Coordinates are meaningful only when valid source georeferencing exists.

### 12.2 Packet sizes and data-reduction arithmetic

For a 3-band 256×256 input tile stored as unsigned 16-bit samples:

\[
3 \times 256 \times 256 \times 2 = 393{,}216\text{ bytes}
\]

| Payload | Bytes | Ratio relative to the 393,216-byte raw tile |
|---|---:|---:|
| Raw three-band input tile | 393,216 | 1× |
| Single-event alert | 48 | 8,192× |
| Six-event packet (maximum documented) | 183 | about 2,148.7× |
| Empty/heartbeat packet | 21 | about 18,724.6× |

The manifest records an aggregate reduction of 6,234.99× across 1,500 simulated frames including heartbeats. This is software/accounting based on the stated traffic model, not a measured RF link, satellite modem, or orbital downlink.

### 12.3 Heartbeat behavior

The model pipeline supports a 21-byte heartbeat every 30 frames to distinguish “no alert” from an inactive processor. The web app intentionally constructs single-tile runs with `heartbeat_enabled=False`, so a fire-free tile can display the zero-fire/no-packet behavior. This demo setting differs from the optional heartbeat-enabled pipeline behavior.

### 12.4 Confidence and contrast interpretation

For Pereira-3C, the event contrast field is a scaled local SWIR2 reflectance contrast—not a thermal-radiance contrast. The confidence field in the checked-in demo is the mean temperature-scaled pixel probability in a component, not the output of the notebook's separate event-level isotonic calibration. These semantics must be retained when the packet is decoded or presented in the UI.

## 13. Downlink queue and prioritization study

The existing technical dossier and README report a **software queue simulation** comparing priority, FIFO, persistence, and retry policies. The figures below are retained as **documented simulation results** from the existing dossier. The current `reports/model_benchmarks/` directory does not contain the separately cited `queue_policy_comparison_v7.csv`, so the detailed numbers are not independently reproducible from a checked-in result CSV at this revision.

| Policy | Simulated capacity | Top-20% urgent events delivered | Mean urgent-event delay | Packets lost permanently | Mean retries |
|---|---|---:|---:|---:|---:|
| Priority + persistence + retry | 50% of required capacity | 87.6% | 4.96 h | 0 | 2.2 |
| FIFO + persistence + retry | 50% of required capacity | 33.6% | 17.87 h | 0 | 2.6 |
| Priority, no persistence/retry | 50% of required capacity | 66.6% | 1.93 h | 910 | 0.0 |
| Priority + persistence + retry | 150% of required capacity | 99.4% | 3.21 h | 0 | 2.2 |
| FIFO + persistence + retry | 150% of required capacity | 87.2% | 6.87 h | 0 | 2.3 |
| Priority, no persistence/retry | 150% of required capacity | 75.5% | 1.93 h | 384 | 0.0 |

Under the reported 50%-capacity scenario, priority + persistence + retry delivers 87.6% of the top-20% urgent events, compared with 33.6% for FIFO + persistence + retry. These are simulated queue-policy results, not measured satellite contact behavior. Realistic orbital-contact geometry, radio capacity, retransmission policy, and onboard persistence must be validated separately.

## 14. Interactive demo: architecture and user workflow

The web application is implemented with **Gradio**. Its purpose is to make the current software pipeline inspectable: the user supplies a bundled sample or a GeoTIFF, chooses inference settings, runs the pipeline, and inspects masks, events, packet bytes, metrics, latency, and data-reduction estimates.

### 14.1 User inputs and controls

The main UI in `demo/app.py` exposes:

- **Preset Gallery:** curated Landsat samples including a large wildfire, medium cluster, one-pixel sensitivity case, fire-free vegetation control, deforestation/fire-front example, and a larger 512×512 scene.
- **Custom GeoTIFF Upload:** accepts a `.tif`/`.tiff` containing either a 10-band Landsat/Pereira patch from which the demo reads bands `[7, 6, 2]`, or a three-band file already ordered `[C7, C6, C2]`.
- **Model variant:** the bundled Pereira-3C INT8 model (~514.8 KiB) or FP32 model (~1.85 MiB).
- **Thread mode:** one-thread or all-available-core inference settings, subject to deployment constraints. The hosted constrained configuration forces one thread.
- **Gate 1 scrub toggle:** enables the isolated-saturation-pixel median scrub.
- **Decision threshold slider:** permits a threshold between 0.50 and 0.99; changing the threshold changes the operating point and must not be mistaken for the fixed benchmark threshold.
- **Execute Edge Pipeline:** runs the selected path and updates the UI outputs.
- **Run 30-Iteration Hardware Benchmark:** measures software latency on the machine hosting the app. It is not a flight-hardware benchmark despite the UI button wording.

The uploaded files are expected to be raw Landsat Level-1-style 16-bit DN. `demo/io_tiff.py` rejects 8-bit imagery and flags inputs whose values look like reflectance or an 8-bit stretch, because they do not match the demo's input contract. This validation is useful but does not replace checks against the full source-product metadata.

### 14.2 Outputs displayed by the demo

| Output | Purpose |
|---|---|
| False-color composite | Displays C7/SWIR2 in red, C6/SWIR1 in green, and C2/Blue in blue, with contrast stretching for visualization |
| Probability heatmap | Visualizes the sigmoid/temperature-scaled pixel score from 0 to 1 |
| Binary segmentation mask | Shows pixels at or above the current threshold |
| Detection overlay | Overlays the predicted mask and event bounding boxes on the false-color image |
| Ground-truth comparison | When a corresponding voting mask exists, compares the result with the Pereira voting reference and reports pixel/event metrics |
| Event and packet JSON | Shows decoded fields and event records for inspection |
| Hex dump | Presents the compact packet bytes in a readable hexadecimal format |
| Protocol dissector | Explains header, event-record, and CRC fields for a ground-station-style view |
| Pipeline telemetry | Reports stage timing, wall time, and available FP32/INT8 mask agreement for the current tile |
| Data-reduction panel | Compares input bytes, lossless zlib size, and encoded alert size |
| Live benchmark | Runs repeated in-process samples and reports latency distribution for the host serving the app |

When a reference mask is available, metrics are explicitly labelled as comparison against the Pereira voting mask—not field ground truth. If no mask is associated with an uploaded/custom scene, the UI reports inference outputs without claiming supervised verification. FP32/INT8 parity is computed on the current frame when both models can be loaded; constrained deployments may skip the side-by-side run to limit memory use.

For large scenes, `demo/io_tiff.py` uses `SceneTiler` to split the image into non-overlapping 256×256 tiles with zero-padding at the image edges and tracks tile offsets for geolocation. The app has a guard for maximum scene size in constrained cloud deployments.

### 14.3 Demo limitations to make clear during a presentation

1. The UI is a software engineering demonstrator, not a live spacecraft, flight computer, RF terminal, or orbital simulator.
2. The checked-in ONNX bundle runs Pereira-3C only. B10-only and hybrid scores are evaluation results reported in the benchmark CSV/manifest; the demo bundle does not contain separate ONNX exports for these configurations.
3. Displayed latency depends on the host, thread count, caching, file I/O, rendering and cloud resource limits. It must not be attributed to a CM4 or spacecraft.
4. A GeoTIFF without a valid affine transform/CRS cannot yield trustworthy event coordinates. The pipeline flags missing geolocation rather than having a sensor-independent source of coordinates.
5. The per-tile demo disables heartbeats by default; the heartbeat behavior is an optional pipeline mode.
6. Single-pixel events are retained (`MIN_EVENT_PIXELS = 1`).
7. The demo's event confidence is not isotonic event-calibrated, as explained in Section 3.

### 14.4 Repository layout for the demo

```text
demo/
├── app.py                    # Gradio interface, controls, orchestration, output panels
├── pipeline.py               # Gate 1, preprocessing, ONNX inference, thresholding,
│                             # connected components, Gate 2, packet encoder/decoder
├── io_tiff.py                # GeoTIFF loading, input-contract checks, CRS conversion,
│                             # geolocation, and non-overlapping scene tiling
├── render.py                 # False-color image, heatmap, mask, overlay, reference comparison
├── metrics.py                # Pixel and event metric calculations for the demo
├── decode_packet.py          # Command-line packet decoder/dissector
├── generate_golden.py        # Generates golden/reference outputs for checks
├── icons.py                  # UI icon and badge helpers
├── models/
│   ├── unet_pereira3c_fp32.onnx
│   ├── unet_pereira3c_int8.onnx
│   ├── manifest_v7.json      # Bands, model IDs, threshold, packet and benchmark metadata
│   └── CHECKSUMS.txt         # Expected model-file checksums
├── sample_data/
│   ├── patches/              # Curated GeoTIFF patches and golden JSON files
│   ├── masks/                # Voting/reference masks used for available comparisons
│   ├── custom_tif/           # Small custom upload examples
│   └── golden.json           # Bundled golden-output/reference metadata
├── requirements.txt          # Gradio, ONNX Runtime, NumPy, SciPy, Rasterio, etc.
├── Dockerfile                # Container image for the app
└── render.yaml               # Deployment configuration
```

The repository's `README.md` also links the live demo at <https://thermal-edge-fire.onrender.com>. Availability and response time depend on the hosting platform, and hosted-cloud latency is not comparable to a controlled benchmark unless the machine and conditions are specified.

### 14.5 Run the demo locally

From the repository root:

```bash
cd demo
pip install -r requirements.txt
python app.py
```

Open the local address printed by Gradio (the documented default is `http://127.0.0.1:7860`). To use Docker:

```bash
docker build -t iastam-edge-demo demo/
docker run -p 7860:7860 --memory="512m" --cpus="1.0" iastam-edge-demo
```

These container limits describe a reproducible demo environment; they are not evidence of spacecraft power or memory performance.

## 15. Known implementation/documentation boundaries

The following points should remain explicit in any technical presentation or later revision:

1. **Radiometric provenance:** the fixed `2e-5 × DN - 0.1` transform is shared by the model path, but the repository documentation should not label it a universal Landsat TOA conversion without verifying the source product and metadata. USGS Level-1 guidance specifies per-band metadata rescaling coefficients and a solar-angle correction for sun-corrected TOA reflectance.
2. **Native versus resampled thermal resolution:** Landsat-8 Band 10 has 100 m native ground resolution and is resampled to a 30 m grid in standard products. A 30 m output grid is not equivalent to 30 m native thermal detail.
3. **Demo event calibration:** the notebook includes an event-confidence calibration report, but `demo/pipeline.py` computes the event field as a mean of temperature-scaled pixel probabilities. Do not claim that the current demo uses event-level isotonic calibration until the code paths are aligned.
4. **Input-shape hardening:** in the current `Gate1.validate`, the leading dimension is read before dimensionality is checked, and the expected channel count is not hard-coded to exactly three. Normal demo loading returns three channels, but malformed external arrays should be rejected by a hardened validator before describing Gate 1 as production-safe.
5. **Gate 2 action:** the greater-than-5% tile-occupancy rule produces a warning, not an automatic rejection.
6. **One-pixel clusters:** `MIN_EVENT_PIXELS = 1`, so the current implementation retains one-pixel components.
7. **Event-metric tolerance:** the notebook matching routine uses a three-pixel spatial tolerance; the metrics are not a strict one-to-one field-validated event assignment.
8. **Queue evidence:** queue figures are simulation results. The current checked-in `reports/model_benchmarks/` folder does not include the detailed queue comparison CSV previously cited by filename.
9. **No flight qualification:** no onboard target-hardware latency, peak RAM, energy per tile, payload sensor characterization, radiation campaign, orbital contact, or real RF-link throughput is established by these files.

## 16. Reproducible result files and entry points

| Repository path | Purpose |
|---|---|
| `notebooks/03_SPACETOON_IASTAM_MODEL_PROTOTYPE_PHASE3.ipynb` | Training/evaluation pipeline, validation selection, robustness and manual-label workflow |
| `demo/models/manifest_v7.json` | Model configuration, calibration, threshold, packet, and stored result metadata |
| `reports/model_benchmarks/test_results_by_config.csv` | Primary held-out test metrics across configurations |
| `reports/model_benchmarks/split_summary.csv` | Train/validation/test label counts |
| `reports/model_benchmarks/manual_zero_shot.csv` | Independent manual-label results |
| `reports/model_benchmarks/benchmark_host.csv` | Model-only host timing statistics |
| `reports/model_benchmarks/pipeline_latency_host.csv` | End-to-end pipeline stage timing statistics |
| `reports/model_benchmarks/robustness_primary.csv` | Synthetic perturbation and Gate 1 scrub results |
| `reports/model_benchmarks/event_confidence_calibration.json` | Event-confidence calibration output from the evaluation workflow; not currently applied by demo `pipeline.py` |
| `demo/app.py` | Gradio application entry point |
| `demo/pipeline.py` | Pipeline runtime, ONNX execution, event extraction and packet protocol |
| `demo/io_tiff.py` | GeoTIFF input, georeferencing and scene tiling |
| `demo/decode_packet.py` | Standalone packet decoder/dissector |

## 17. Conclusions and next validation steps

The repository supports describing Thermal-Edge-Fire as a **validated software information-extraction prototype on Landsat/Pereira data, with a proposed 6U-class edge-deployment architecture**. It supports reporting the Pereira-3C test metrics, the manual-label zero-shot results, host timing, synthetic robustness tests, compact packet arithmetic, and queue simulation—with the caveats defined above.

Before making stronger operational or flight claims, the next engineering steps are to:

1. validate the normalization against the exact source-product metadata and lock the input contract;
2. harden the input-shape/type checks and add tests for malformed inputs and missing georeferencing;
3. align notebook and demo event-confidence semantics or clearly version the difference;
4. test the exported models and full pipeline on the target embedded hardware and record latency, peak memory, and energy;
5. evaluate the intended thermal payload with thermal-native labels or a validated simulated-raw sensor model; and
6. validate packet delivery/queue behavior with a concrete contact schedule and communications budget.

Until those tests are performed, **“final production model,” “flight proven,” “deterministic detection,” “radiation tolerant,” “fits entirely in onboard SRAM,” and “zero future false alarms” are not supported descriptions** of the current implementation.

## 18. References

- Repository README and scope statement: <https://github.com/kacemamine721-star/thermal-edge-fire/blob/main/README.md>
- Model manifest: <https://github.com/kacemamine721-star/thermal-edge-fire/blob/main/demo/models/manifest_v7.json>
- USGS, *Using the USGS Landsat Level-1 Data Product* (band-specific metadata factors and TOA conversion): <https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product>
- USGS, *What are the band designations for the Landsat satellites?* (spectral bands and ground resolution, including Band 10): <https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites>
- USGS, *Landsat Collection 2 Level-1 Data* (Level-1 DN product and metadata): <https://www.usgs.gov/landsat-missions/landsat-collection-2-level-1-data>