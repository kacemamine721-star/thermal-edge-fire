# Gate 2: Post-Treatment Output & Physical Sanity Gate Review

## Scope

Gate 2 validates preprocessed model inputs and fire predictions before an alert
is accepted for downlink. The implementation is in
`src/fireedge/validation/output_validator.py`.

The processing order is:

1. Preprocessing sanity.
2. Catastrophic prediction collapse check.
3. Spatial component filtering.
4. Local thermal consistency.
5. Alert packet construction and verification.

The first four stages are orchestrated by `validate_output()`. Packet creation
and packet verification are separate calls because geolocation and satellite
footprint data are external to the image-validation function.

## Work Completed

### 1. Preprocessing sanity

Implemented by `check_preprocessing_sanity()`:

- Rejects NaN and Inf values above the configured 1% fraction.
- Checks finite values against the configured physical LWIR brightness
  temperature range, defaulting to 180 K through 400 K.
- Rejects frames with more than the configured 1% out-of-range fraction.
- Calculates a subnormal/denormal fraction and includes it in the result detail.
- Returns a structured `ValidationResult` with a specific `RejectReason`.

### 2. Catastrophic model collapse detection

Implemented by `check_spatial_plausibility()`:

- Computes the predicted fire area fraction before expensive processing.
- Rejects predictions above 50% of the frame as
  `CATASTROPHIC_COLLAPSE`.
- Sets `telemetry_flag=True` for this condition so operations can distinguish a
  likely model or pipeline failure from an ordinary no-fire result.

### 3. Spatial plausibility

Also implemented by `check_spatial_plausibility()`:

- Uses 8-connected component labeling.
- Rejects a prediction when all candidate components are smaller than the
  configured minimum, defaulting to two pixels.
- Removes isolated one-pixel blips before thermal analysis.
- Caps the number of components entering the expensive thermal stage at 64,
  retaining the largest components when the cap is exceeded.

### 4. Thermal consistency

Implemented by `check_thermal_consistency()`:

- Builds a local background ring around each connected prediction.
- Compares the median candidate temperature with the median ring temperature.
- Requires the candidate to be hotter than the local background by both a
  minimum temperature margin and a configurable local-noise margin. The
  effective requirement is the stricter of the two.
- Rejects cold cloud edges and similar cold false alarms as
  `THERMAL_INCONSISTENT`.
- Supports an optional coastal mask with an additional temperature margin.
- Uses the candidate median rather than its maximum to reduce sensitivity to a
  single hot outlier.

### 5. Downlink packet verification

Implemented by `AlertPacket`, `pack_alert_packet()`, and
`validate_alert_packet()`:

- Encodes schema version, timestamp, latitude, longitude, native-frame
  bounding box, confidence, intensity, flags, and optional crop bytes.
- Uses a fixed 27-byte little-endian header and a 4-byte CRC-32 trailer.
- Enforces the configured maximum packet size, defaulting to 200 bytes.
- Verifies CRC before accepting the packet.
- Validates schema version, crop length, geographic coordinate bounds, positive
  bounding-box dimensions, confidence quantization, and reserved flags.
- Checks coordinates against an optional fast bounding box and then the full
  satellite footprint polygon.
- Provides `crc16_ccitt()` as a smaller-trailer alternative for a future flight
  packet format.

### 6. Integration cleanup completed

- Repaired `src/fireedge/validation/__init__.py`, which referenced missing
  `OutputValidator`, `OutputValidatorConfig`, and `OutputValidationReport`
  symbols from an older API.
- Exported the implemented functional Gate 2 API and aliases:
  `OutputValidatorConfig`, `OutputValidationReport`, `RejectReason`,
  `validate_output()`, packet helpers, and stage-level checks.
- Replaced the stale Gate 2 tests that expected DN-scale values and the removed
  class-based API with tests for the current Kelvin-based functional API.
- Added a regression test for invalid zero-sized packet bounding boxes.

## Verification Evidence

The Gate 2 test coverage now includes:

- Clean preprocessing pass.
- Excessive non-finite values.
- Excessive physical-range violations.
- Isolated one-pixel prediction rejection.
- Greater-than-50% catastrophic collapse rejection and telemetry flagging.
- Hot candidate acceptance.
- Cold candidate rejection.
- Valid packet CRC, schema, size, and footprint acceptance.
- Checksum corruption rejection.
- Out-of-footprint coordinate rejection.
- Invalid zero-sized bounding-box rejection.

The focused test module contains 11 tests. In this environment, `pytest` is
not installed, so the tests were compiled and each test function was executed
directly with the repository `src` path configured. All 11 passed.

## Limitations and Open Risks

1. **The main orchestration does not validate a packet.** `validate_output()`
   stops after thermal consistency. The caller must explicitly construct and
   validate the packet before queueing it for downlink. That call boundary must
   be enforced by the production pipeline.

2. **Inverted dynamic ranges are not explicitly modeled.** The current sanity
   check validates physical value bounds, but it does not receive min/max
   metadata for a tensor or verify that a separate normalization range is
   ordered. An inverted normalization configuration could therefore be missed.

3. **Subnormal values are informational only.** The configured
   `max_subnormal_fraction` is not used as a rejection or flag threshold;
   subnormal prevalence is only reported in the detail string.

4. **Input shapes and alignment are assumed.** The functions do not provide a
   dedicated shape contract for the tensor, thermal frame, fire mask, or
   coastal mask. Mismatched arrays can raise NumPy/SciPy errors instead of
   returning a structured rejection.

5. **The thermal rule is a heuristic, not a fire classifier.** A hot sunlit
   roof, bare rock, industrial heat source, or warm sensor artifact can satisfy
   the local temperature test. The coastal margin is only a partial mitigation
   and depends on a correctly aligned coastal mask.

6. **The background ring has edge cases.** A candidate touching the frame edge
   can have no usable ring and is conservatively not confirmed. Large or dense
   candidate regions can also make the local ring a poor estimate of true
   background.

7. **Component capping drops candidates silently.** If more than 64 components
   survive spatial filtering, only the largest components reach thermal
   analysis. The result does not currently report which candidates were
   dropped or set a telemetry flag for the cap.

8. **Packet construction and validation are separate responsibilities.** The
   validator checks packet fields after packing, but `pack_alert_packet()` does
   not itself return a structured validation result for invalid confidence,
   coordinates, dimensions, crop sizes, or values that cannot fit their binary
   fields.

9. **The packet format is a prototype contract.** It uses CRC-32 despite the
   module also providing CRC-16, and it does not yet define endianness/version
   negotiation with the ground segment, replay protection, authentication, or
   a formal CCSDS wrapper.

10. **Footprint geometry is simplified.** The dependency-free ray-casting
    polygon check does not document behavior for antimeridian-crossing
    footprints, polar geometry, self-intersecting polygons, or invalid polygon
    input.

11. **Physical units must be enforced by integration.** The current defaults
    expect brightness temperature in Kelvin. Passing raw DN values, normalized
    `[0, 1]` tensors, or mixed units will be rejected or misinterpreted unless
    the caller converts and contracts the data correctly.

12. **Flight performance is not yet demonstrated.** No benchmark in this
    repository establishes latency, memory use, or deterministic behavior on the
    target onboard processor. SciPy connected-component labeling is suitable
    for the ground prototype but may need a bounded embedded implementation.

## Remarks / Required Next Checks

- Add a production-facing Gate 2 orchestration test that cannot queue a packet
  unless both `validate_output()` and `validate_alert_packet()` pass.
- Add explicit shape, dtype, unit, and mask-alignment checks with structured
  rejection reasons.
- Add an explicit normalization/dynamic-range contract that rejects inverted or
  inconsistent ranges.
- Decide whether subnormal prevalence should remain informational or become a
  configurable flag/rejection.
- Report component-cap truncation and preserve the dropped-component count in
  telemetry metrics.
- Add field-level packet tests for crop-length mismatch, unsupported schema
  version, reserved flags, invalid confidence, and malformed footprint input.
- Specify the flight packet protocol with the ground segment, including CRC
  choice, versioning, authentication, replay handling, and CCSDS framing.
- Validate the thermal thresholds against representative post-treatment output
  and hard negatives: cold cloud edges, coastlines, sunlit rock, cities, and
  industrial heat sources.
- Benchmark a SciPy-free implementation on the target processor before using
  the current prototype in flight software.

## Gate 2 Assessment

**Implementation status: functionally implemented, package-integrated, and
covered by focused tests for the specified synthetic scenarios.**

**Readiness status: ground-prototype ready, not yet flight- or production-ready.**
The most important remaining control is enforcing packet validation after
`validate_output()` and before downlink queue insertion, followed by unit
contracts, real post-treatment data calibration, formal packet definition, and
target-hardware benchmarking.