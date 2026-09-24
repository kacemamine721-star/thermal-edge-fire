# Gate 1: Pre-Treatment Input Quality Gate Review

## Scope

Gate 1 validates raw detector frames before NUC, calibration, normalization, or
CNN inference. The implementation is located under `src/fireedge/validation/`.
It returns one of three outcomes:

- `PASS`: no configured quality issue was found.
- `PASS_WITH_FLAGS`: the frame can continue, but a non-fatal anomaly was recorded.
- `REJECT`: the frame must be discarded before treatment.

The public entry points are `GroundGateValidator` for curation/ground data and
`FlightGateValidator` for single-channel orbital inference. `FrameValidator` is
currently a backward-compatible alias for `GroundGateValidator`.

## Work Completed

### Universal checks

Implemented in `base_validator.py` and applied by both modes:

- Shape and minimum-size validation.
- NaN/Inf detection. Default rejection threshold is greater than 1% of all
	pixels (`nonfinite_frac_reject=0.01`).
- No-data detection. Default rejection threshold is greater than 30% of all
	pixels (`nodata_frac_reject=0.30`, with `0` as the default no-data value).
- Dead/stuck frame detection using raw valid-pixel standard deviation.
- High-frequency noise estimation using a 3x3 median residual and robust MAD
	sigma. Calibrated configurations can flag or reject excessive noise.
- Saturation detection against the configured ADC limit. The default is 16-bit
	saturation at 65535, with rejection above 2% of valid pixels.
- Configurable physical range checks (`valid_min`/`valid_max`).
- Non-physical neighbor-gradient checks using calibrated `max_gradient`.
- Defective row/column striping detection using robust line-profile residuals.
- Hot-pixel candidate and isolation detection, with optional median repair.
- Metrics and reason codes are returned in `ValidationReport`; batch validation
	also reports pass, flagged, and rejected frame totals.

### Mode A: Ground/Curation

Implemented in `ground_gate_validator.py`:

- Runs the universal checks independently per channel.
- Supports multi-channel thermal data such as Landsat-8 B10/B11.
- Cross-channel confirmation suppresses isolated hot-pixel rejection when a
	corresponding anomaly is also detected in another channel.
- Preserves confirmed anomalies for recall and reports confirmation metrics.
- Can optionally replace suspected isolated hot pixels with a local 3x3 median
	when `hot_pixel_policy="repair"`.

### Mode B: Flight/Orbital Inference

Implemented in `flight_gate_validator.py`:

- Operates on a 2D single-channel frame or the first channel of a 3D input.
- Uses local residuals and neighboring pixels to identify hot candidates.
- Applies a PSF-inspired halo ratio and connected-neighbor test.
- Flags isolated, high-value, no-halo candidates as `SEU_TRANSIENTS`.
- Preserves candidates with neighboring elevation as optical candidates.
- Supports optional repair of flagged transients using a local 3x3 median.

## Verification Evidence

The Gate 1 tests cover:

- Clean-frame pass behavior.
- Dead/stuck constant frames.
- Excessive saturation.
- Excessive no-data.
- Isolated hot pixels.
- Same-location cross-channel confirmation.
- Defective-column striping.
- Empirical threshold calibration from clean frames.
- Flight-mode SEU detection.
- Flight-mode blurred optical hotspot preservation.
- Flight-mode transient repair.

These are synthetic fault-injection tests. They verify algorithm behavior, not
performance on a representative raw telemetry corpus.

## Limitations and Open Risks

1. **Threshold calibration is not operationally connected.** The validator
	 accepts calibrated limits, and `calibrate_from_frames()` can derive several
	 limits, but the repository does not yet show a validated calibration artifact
	 being generated from real sensor data and loaded into the production path.

2. **The configured YAML is not automatically enforced by these validators.**
	 The validators receive a `ValidatorConfig` object directly. A caller must
	 explicitly map `configs/config.yaml` into that object; otherwise Python
	 defaults are used.

3. **Mode A does not enforce the B10/B11 contract.** It accepts any 3D array
	 with at least three rows and columns and confirms against any other channel.
	 It does not select channels 8 and 9, verify band metadata, or require exactly
	 the Landsat-8 B10/B11 pair. The caller must select and align those bands.

4. **Cross-channel confirmation is a permissive heuristic.** It checks whether
	 another channel exceeds a reduced hot-pixel z threshold at the same pixel,
	 but does not account for registration error, differing band response, cloud
	 effects, or radiometric uncertainty.

5. **The PSF check is not instrument-calibrated.** `halo_min_ratio` is a fixed
	 configurable heuristic. There is no payload-specific PSF kernel, modulation
	 transfer function, ground-truth SEU corpus, or orbit validation establishing
	 that the selected ratio separates optical hotspots from radiation events.

6. **The flight validator silently uses channel 0 for multi-channel input.**
	 This is convenient for compatibility but can hide an integration error. A
	 production flight interface should reject unexpected channel counts or make
	 the primary channel explicit.

7. **The halo metric is a maximum-neighbor ratio, not an energy-based PSF
	 measurement.** A candidate with any hot neighbor can be treated as optical,
	 even when the neighborhood does not match the actual optical blur profile.

8. **No-data and non-finite fractions use the full frame as denominator.** This
	 matches the current gate semantics, but it should remain an explicit contract
	 when partial readouts, masks, or overlapping invalid pixels are introduced.

9. **Repair is optional and not provenance-preserving by itself.** Repaired
	 pixels are returned in `ValidationReport.cleaned`, but there is no built-in
	 audit record containing the original values, repair count, or downstream
	 approval decision.

10. **Performance and resilience claims are unverified here.** The repository
		does not currently provide benchmark evidence for the documented onboard
		latency, memory usage, integer-only execution, or behavior under malformed
		arrays and extreme numeric ranges.

## Remarks / Required Next Checks

- Add an integration test that loads the YAML validation settings and confirms
	they are the settings used by both Gate 1 modes.
- Add a real-data calibration report containing frame provenance, derived
	thresholds, distributions, and acceptance rationale.
- Add a Landsat adapter or explicit input contract that extracts and validates
	B10/B11, including spatial co-registration and missing-band failures.
- Obtain representative raw CubeSat/microbolometer frames and label SEUs,
	optical hotspots, striping, saturation, and dropped lines for threshold
	tuning.
- Calibrate the flight halo threshold from the actual optical PSF and evaluate
	false-positive and false-negative rates across hotspot sizes and signal levels.
- Add boundary tests for exactly 1% non-finite, exactly 30% no-data, exactly 2%
	saturation, empty/near-empty frames, invalid channel counts, and edge pixels.
- Decide whether repair is allowed before preprocessing. If it is, persist the
	original frame reference, repair mask, reason codes, and replacement counts.
- Benchmark Gate 1 on the target processor and record latency by check and by
	frame size before treating the latency claim as acceptance evidence.

## Gate 1 Assessment

**Implementation status: functionally implemented and unit-tested for the
specified synthetic scenarios.**

**Readiness status: not yet flight- or production-ready.** The main remaining
work is sensor-specific calibration and integration: enforce the B10/B11 input
contract in ground mode, validate the PSF model on real single-channel payload
data, wire configuration loading, and establish operational evidence from real
raw telemetry.
