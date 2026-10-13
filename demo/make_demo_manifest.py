#!/usr/bin/env python3
"""
Build models/manifest.json (demo manifest, no internal version label) from the notebook's manifest_v7.json,
then rewrite models/CHECKSUMS.txt so the SHA-256 test still passes.

Usage (from the demo folder):
    python make_demo_manifest.py                     # reads models/manifest_v7.json
    python make_demo_manifest.py path/to/manifest_v7.json

Only labels and text are changed. Every number (threshold, temperature, packet sizes, results) is copied as is.
After running it you may delete models/manifest_v7.json (pipeline.py reads manifest.json first).
"""
import hashlib
import json
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
MODELS = DEMO_DIR / "models"
src = Path(sys.argv[1]) if len(sys.argv) > 1 else MODELS / "manifest_v7.json"

m = json.loads(src.read_text(encoding="utf-8"))
m["version"] = "multiband information-product pipeline (Pereira-3C payload proxy)"
m["limitations"] = [s.replace("V7 reports", "the report states").replace("V7", "this pipeline")
                    for s in m.get("limitations", [])]
for k, v in list(m.get("artifacts", {}).items()):          # drop notebook-side paths (Kaggle working directory)
    m["artifacts"][k] = Path(v).name
dumped = json.dumps(m, indent=2, ensure_ascii=False) + "\n"
assert "V7" not in dumped and "v7" not in dumped, "a 'V7' string is still present"
out = MODELS / "manifest.json"
out.write_text(dumped, encoding="utf-8", newline="\n")
print("Wrote", out)

# CHECKSUMS.txt: same normalisation as test_demo.py (CRLF -> LF for .json/.txt)
lines = ["# SHA-256 of the files in models/ (CRLF normalised to LF for .json and .txt)"]
for p in sorted(MODELS.iterdir()):
    if p.name in ("CHECKSUMS.txt", "manifest_v7.json") or not p.is_file():
        continue
    data = p.read_bytes()
    if p.suffix in (".json", ".txt"):
        data = data.replace(b"\r\n", b"\n")
    lines.append(f"{hashlib.sha256(data).hexdigest()}  {p.name}")
(MODELS / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("Wrote", MODELS / "CHECKSUMS.txt")
print("\n".join(lines[1:]))
