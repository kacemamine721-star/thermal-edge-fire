"""
CRITICAL THINKING & EDGE CASES (per check)
-------------------------------------------------------------------------
1. Preprocessing sanity
   - Breaks if a whole tensor is shifted (e.g. calibration LUT off-by-one)
     but still finite and in-range: this check cannot catch a plausible-
     looking but wrong calibration, only outright corruption.
   - A genuine, very small (sub-pixel) fire early in ignition may run
     right at the edge of the physical-range floor on a cold background --
     the range check should never be tightened to the point it clips real
     low-intensity smoldering signatures; validate the chosen bounds
     against real fire-onset radiometry, not just "typical Earth scene."

2. Spatial plausibility
   - The >50% catastrophic-collapse threshold is a blunt instrument: a
     genuine, very large active megafire complex (real-world events have
     covered enormous fractions of a sensor's field of view) could in
     principle be legitimately large. This is the one check most likely to
     suppress a true positive at scale -- recommend logging (not silently
     discarding) any collapse event with telemetry_flag=True so ground ops
     can distinguish "model broke" from "the fire is genuinely huge" on
     the next pass, rather than losing the event entirely.
   - min_component_pixels as a flat threshold ignores that a real distant
     small fire and an SEU can both be 1px at certain ranges/resolutions;
     this check is only as good as Gate 1's PSF-halo discrimination feeding
     it a mask that's already been through that filter. Treat this as
     defense-in-depth, not the primary discriminator.

3. Thermal consistency
   - Sun glint is the hardest failure mode this gate does not fully solve:
     specular reflection can be genuinely hot/bright in-band, not just
     high-contrast, so the signed-delta check alone will not reject it.
     Partial mitigations: known sun-angle/viewing-geometry exclusion zones
     (predictable glint geometry), or a second independent band -- neither
     of which this single-channel LWIR gate has natively.
   - Coastal false-triggers persist on hot, sunlit rock/sand at midday
     even with `coastal_mask` margin -- this is a real, only partially
     mitigated limitation, not a solved case.
   - A true, very high-intensity active megafire can saturate the local
     background ring itself (heat plume affecting "background" pixels
     too), compressing the computed delta and risking a false reject of
     a genuine large fire. Consider a saturation-aware fallback (e.g. if
     bg_std spikes anomalously, widen the ring or fall back to a broader
     reference baseline).

4. Downlink packet verification
   - Geolocation sanity is only as good as the ephemeris/attitude solution
     feeding it -- a systematic pointing error will pass this check while
     being wrong, since it's internally self-consistent.
   - CRC catches transmission/storage corruption, not encoding-logic bugs
     upstream of packing -- unit-test pack/unpack round-trips separately.

LIMITATIONS & HARDWARE CONSTRAINTS
-------------------------------------------------------------------------
- This module is a ground-prototype reference implementation (numpy/scipy,
  Python). It is explicitly NOT flight software as written: scipy is a
  heavy dependency unlikely to be available on a SWaP-constrained onboard
  processor. A flight port would reimplement `ndimage.label` /
  `binary_dilation` as lightweight fixed-point C/C++ routines (or use a
  vetted flight-heritage image-processing library) -- budget real
  engineering time for this rewrite, it is not a drop-in swap.
- Connected-component labeling and windowed thermal stats are the most
  compute- and memory-intensive stages; `max_candidate_components` exists
  specifically to bound stage 4's worst-case cost on a pathological frame,
  but the right cap value must be profiled against actual onboard CPU/RAM
  budget, not assumed.
- Every threshold in `ValidatorConfig` (deltas, fractions, ring size) is a
  starting point, not a validated flight value -- they must be tuned
  against real sensor NETD, actual optics PSF characterization, and your
  THRawS domain-gap results before being trusted operationally.
- This gate assumes Gate 1 already ran; it does not re-validate raw sensor
  telemetry from scratch, only the model's output and the preprocessing
  that fed it.
"""