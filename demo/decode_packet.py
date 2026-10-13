#!/usr/bin/env python3
"""
Ground-station decoder for the compact binary alert packet (version 3)
======================================================================
Decodes a raw hex string or a binary packet file and prints a field-by-field dissection.

Usage:
    # 1. Decode a hex string given on the command line:
    python decode_packet.py --hex "<hex string of the packet>"

    # 2. Decode a binary packet file:
    python decode_packet.py --file packet.bin

    # 3. Machine-readable output:
    python decode_packet.py --file packet.bin --json

    # 4. Interactive mode (press Enter to decode a built-in 48-byte sample packet):
    python decode_packet.py
"""

import argparse
import json
import sys
from pathlib import Path

# Add the demo directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
from pipeline import decode_packet, HEADER_BYTES, EVENT_BYTES, EVENT_FMT, HEADER_FMT  # noqa: E402,F401

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_NAMES = {1: "pereira3c (C7+C6+C2)", 2: "b10 (thermal control)", 3: "pereira3c_b10 (hybrid)"}
HEADER_FLAG_NAMES = {1: "INPUT_REJECTED", 2: "OUTPUT_REJECTED", 4: "INPUT_REPAIRED (scrub / dead columns)",
                     8: "SATURATED_INPUT", 16: "GATE2_WARN", 32: "HEARTBEAT", 64: "TIME_UNAVAILABLE",
                     128: "REFERENCE_ONLY"}
EVENT_FLAG_NAMES = {1: "NO_GEO", 2: "GEO_UNCERT_UNKNOWN", 4: "SWIR_METRIC", 8: "THERMAL_METRIC"}


def _names(value: int, table: dict) -> str:
    return ", ".join(n for b, n in table.items() if int(value) & b) or "none"


def print_dissector(raw_bytes: bytes):
    """Prints a byte-by-byte dissection of one packet."""
    decoded = decode_packet(raw_bytes)
    total_len = len(raw_bytes)
    crc_ok = bool(decoded.get("crc_valid"))
    len_ok = bool(decoded.get("length_valid", True))
    crc_str = "[VALID] CRC matches" if crc_ok else "[CORRUPTED] CRC mismatch"
    ts = int(decoded.get("timestamp_ms", 0))

    print("\n" + "=" * 80)
    print(" ALERT PACKET DISSECTOR (version 3)")
    print("=" * 80)
    print(f" Total Packet Size : {total_len} bytes")
    print(f" CRC-32 Checksum   : {decoded.get('crc32', 'N/A')} {crc_str}")
    if not len_ok:
        print(" LENGTH CHECK      : [ERROR] byte count does not match the event count in the header")
    mid = int(decoded.get("model_id", 0))
    print(f" Model ID          : {mid} ({MODEL_NAMES.get(mid, 'unknown model id')})")
    print(f" Frame Sequence    : #{decoded.get('frame_id', 0)}")
    print(f" Acquisition time  : {ts} ms" if ts else " Acquisition time  : 0 (not available, see TIME_UNAVAILABLE flag)")
    print(f" Events In Packet  : {decoded.get('n_events', 0)}")
    if decoded.get("events_dropped"):
        n_drop = int(decoded["events_dropped"])
        print(f" Events dropped    : {n_drop}{'+' if n_drop >= 15 else ''} (more events than the packet can hold; "
              f"the most confident are sent)")
    print("-" * 80)

    print(f"\n[*] HEADER ({HEADER_BYTES} bytes):")
    flags = raw_bytes[16]
    header_fields = [
        (0, 1, "Magic", "0x" + raw_bytes[0:1].hex(), "0xF1 = alert packet"),
        (1, 2, "Protocol Version", "0x" + raw_bytes[1:2].hex(), f"Version {raw_bytes[1]}"),
        (2, 3, "Model ID", "0x" + raw_bytes[2:3].hex(), f"ID {raw_bytes[2]} ({MODEL_NAMES.get(raw_bytes[2], 'unknown')})"),
        (3, 7, "Frame Sequence ID", "0x" + raw_bytes[3:7].hex(), f"Frame #{int.from_bytes(raw_bytes[3:7], 'little')}"),
        (7, 15, "Acquisition time", "0x" + raw_bytes[7:15].hex(),
         f"{ts} ms" if ts else "0 = not available"),
        (15, 16, "Event Count", "0x" + raw_bytes[15:16].hex(), f"{raw_bytes[15]} event(s) in packet"),
        (16, 17, "Packet Flags", "0x" + raw_bytes[16:17].hex(), f"{flags}: {_names(flags, HEADER_FLAG_NAMES)}"),
    ]
    for start, end, name, hx, val in header_fields:
        print(f"  [{start:03d}..{end - 1:03d}] {name:<18} | Raw: {hx:<20} | {val}")

    for i, ev in enumerate(decoded.get("events", []), 1):
        ev_offset = HEADER_BYTES + (i - 1) * EVENT_BYTES
        print(f"\n[+] EVENT #{i} (bytes {ev_offset}..{ev_offset + EVENT_BYTES - 1}, {EVENT_BYTES} bytes):")
        if ev['flags'] & 1:
            print("  * Location     : not available (NO_GEO flag)")
        else:
            print(f"  * Location     : lat {ev['lat']:.5f}, lon {ev['lon']:.5f} (WGS84)")
        print(f"  * Bounding Box : x={ev['x']}, y={ev['y']}, w={ev['w']}, h={ev['h']} (pixels)")
        print(f"  * Area         : {ev['area_ha']:.1f} ha (stored in units of 0.1 ha)")
        print(f"  * Confidence   : {ev['confidence'] * 100.0:.2f}% (mean temperature-scaled pixel probability)")
        print(f"  * SWIR2 contrast: {ev['metric_u16'] / 10000.0:.3f} (C7 reflectance, event p95 minus local ring median)")
        print(f"  * Pixel size   : {ev['gsd_m_u16']} m")
        unc = ev['geo_uncertainty_m_u16']
        print(f"  * Location error: {'unknown' if unc >= 65535 else str(unc) + ' m'}")
        print(f"  * Event flags  : 0x{ev['flags']:02X} ({_names(ev['flags'], EVENT_FLAG_NAMES)})")

    crc_offset = total_len - 4
    print(f"\n[*] CRC-32 (bytes {crc_offset}..{total_len - 1}, 4 bytes):")
    print(f"  * Stored value : 0x{raw_bytes[crc_offset:][::-1].hex()} (little-endian on the wire: {raw_bytes[crc_offset:].hex()})")
    print(f"  * Result       : {crc_str}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Alert packet decoder")
    parser.add_argument("--hex", type=str, help="Hexadecimal string of the packet")
    parser.add_argument("--file", type=str, help="Path to a raw binary packet file (.bin)")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the dissection")
    args = parser.parse_args()

    raw_bytes = None
    if args.hex:
        raw_bytes = bytes.fromhex(args.hex.strip().replace(" ", "").replace("\n", ""))
    elif args.file:
        with open(args.file, "rb") as f:
            raw_bytes = f.read()
    else:
        print("--- Alert packet decoder ---")
        val = input("Paste hex string (or press Enter to decode a built-in sample packet): ").strip()
        if val:
            raw_bytes = bytes.fromhex(val.replace(" ", ""))
        else:
            from pipeline import encode_packet
            sample_ev = [{
                'lat_e7': 368000000, 'lon_e7': 101800000,
                'x': 120, 'y': 85, 'w': 2, 'h': 2,
                'area_ha_x10': 3, 'confidence_u16': 64000, 'metric_u16': 2500,
                'gsd_m_u16': 30, 'geo_uncertainty_m_u16': 65535, 'flags': 6
            }]
            raw_bytes = encode_packet(1, 1, 1728500000000, sample_ev, flags=0)

    try:
        if args.json:
            print(json.dumps(decode_packet(raw_bytes), indent=2))
        else:
            print_dissector(raw_bytes)
    except ValueError as exc:
        print(f"[ERROR] Not a valid alert packet: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()