#!/usr/bin/env python3
"""
IASTAM Ground Station — Compact Binary Alert Packet (V3) Decoder CLI
===================================================================
A standalone CLI tool to decode raw hex or binary alert packets for the jury.

Usage:
    # 1. Decode hex string directly from command line:
    python decode_packet.py --hex "f1030100000001000001815b80a6800100..."

    # 2. Decode binary packet file:
    python decode_packet.py --file packet.bin

    # 3. Interactive prompt mode:
    python decode_packet.py
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add demo directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
from pipeline import decode_packet, HEADER_FMT, EVENT_FMT, HEADER_BYTES, EVENT_BYTES

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def print_dissector(raw_bytes: bytes):
    """Prints a clean byte-by-byte ground station wire dissection for the jury."""
    decoded = decode_packet(raw_bytes)
    total_len = len(raw_bytes)

    print("\n" + "=" * 80)
    print(" [SATELLITE] IASTAM MISSION GROUND TELEMETRY -- PACKET PROTOCOL DISSECTOR (V3)")
    print("=" * 80)
    print(f" Total Packet Size : {total_len} Bytes")
    crc_str = "[VALID] (Integrity Confirmed)" if decoded.get("crc_valid") else "[CORRUPTED] CRC Mismatch"
    print(f" CRC-32 Checksum   : {decoded.get('crc32', 'N/A')} ({crc_str})")
    print(f" Model ID          : {decoded.get('model_id', 1)} (IASTAM Tiny U-Net SWIR3)")
    print(f" Frame Sequence    : #{decoded.get('frame_id', 0)}")
    print(f" Timestamp (ms)    : {decoded.get('timestamp_ms', 0)}")
    print(f" Events In Packet  : {decoded.get('n_events', 0)}")
    print("-" * 80)

    # Dissect Header (Bytes 0-16)
    print("\n[*] MISSION TELEMETRY HEADER (17 Bytes):")
    header_fields = [
        (0, 1, "Magic Sync Byte", "0x" + raw_bytes[0:1].hex(), "0xF1 (Active Fire Alert Packet)"),
        (1, 2, "Protocol Version", "0x" + raw_bytes[1:2].hex(), f"Version {raw_bytes[1]}"),
        (2, 3, "Model Variant ID", "0x" + raw_bytes[2:3].hex(), f"ID {raw_bytes[2]} (pereira3c SWIR3)"),
        (3, 7, "Frame Sequence ID", "0x" + raw_bytes[3:7].hex(), f"Frame #{int.from_bytes(raw_bytes[3:7], 'little')}"),
        (7, 15, "Epoch Timestamp", "0x" + raw_bytes[7:15].hex(), f"{int.from_bytes(raw_bytes[7:15], 'little')} ms"),
        (15, 16, "Event Count", "0x" + raw_bytes[15:16].hex(), f"{raw_bytes[15]} events in packet"),
        (16, 17, "Pipeline Flags", "0x" + raw_bytes[16:17].hex(), f"Flags = {raw_bytes[16]} (0=Nominal)")
    ]
    for start, end, name, hx, val in header_fields:
        print(f"  [{start:03d}..{end-1:03d}] {name:<22} | Raw: {hx:<18} | Value: {val}")

    # Dissect Events (27 bytes each)
    for i, ev in enumerate(decoded.get("events", []), 1):
        ev_offset = 17 + (i - 1) * 27
        print(f"\n[+] EVENT RECORD #{i} (Bytes {ev_offset}..{ev_offset + 26} | 27 Bytes):")
        print(f"  * Geolocation  : Lat {ev['lat']:.5f} deg, Lon {ev['lon']:.5f} deg (WGS84 EPSG:4326)")
        print(f"  * Bounding Box : Pixel (x={ev['x']}, y={ev['y']}, w={ev['w']}, h={ev['h']})")
        print(f"  * Affected Area: {ev['area_ha']:.2f} Hectares ({ev['area_ha_x10']/10.0:.1f} ha)")
        print(f"  * Confidence   : {ev['confidence']*100.0:.2f}% (calibrated posterior probability)")
        print(f"  * SWIR2 Metric : Delta-rho = {ev['metric_u16']/10000.0:.3f} (radiant thermal contrast)")
        print(f"  * Sensor GSD   : {ev['gsd_m_u16']} meters/pixel")
        print(f"  * Event Flags  : 0x{ev['flags']:02X}")

    # Dissect CRC32
    crc_offset = total_len - 4
    crc_hex = "0x" + raw_bytes[crc_offset:].hex()
    print(f"\n[*] CRC-32 CHECKSUM (Bytes {crc_offset}..{total_len-1} | 4 Bytes):")
    print(f"  * Checksum Bytes: {crc_hex} (IEEE 802.3 standard)")
    print(f"  * Verified Value: {decoded.get('crc32', 'N/A')} -> {crc_str}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="IASTAM Binary Alert Packet Decoder for Jury Demonstration")
    parser.add_argument("--hex", type=str, help="Hexadecimal string of the binary packet")
    parser.add_argument("--file", type=str, help="Path to raw binary packet file (.bin)")
    parser.add_argument("--json", action="store_true", help="Output pure JSON instead of dissector table")

    args = parser.parse_args()

    raw_bytes = None
    if args.hex:
        raw_bytes = bytes.fromhex(args.hex.strip().replace(" ", "").replace("\n", ""))
    elif args.file:
        with open(args.file, "rb") as f:
            raw_bytes = f.read()
    else:
        # Prompt user
        print("--- IASTAM Interactive Packet Decoder ---")
        val = input("Paste Hex String (or press Enter to decode default test packet): ").strip()
        if val:
            raw_bytes = bytes.fromhex(val.replace(" ", ""))
        else:
            # Generate a sample 48-byte packet
            from pipeline import encode_packet
            sample_ev = [{
                'lat_e7': 368000000, 'lon_e7': 101800000,
                'x': 120, 'y': 85, 'w': 2, 'h': 2,
                'area_ha_x10': 3, 'confidence_u16': 64000, 'metric_u16': 2500,
                'gsd_m_u16': 30, 'geo_uncertainty_m_u16': 65535, 'flags': 4
            }]
            raw_bytes = encode_packet(1, 1, 1728500000000, sample_ev, flags=0)

    if args.json:
        res = decode_packet(raw_bytes)
        print(json.dumps(res, indent=2))
    else:
        print_dissector(raw_bytes)


if __name__ == "__main__":
    main()
