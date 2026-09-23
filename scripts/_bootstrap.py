"""Bootstrap sys.path for standalone scripts in thermal-edge-fire."""
import os
import sys
from pathlib import Path

# Add <root>/src to sys.path
_scripts_dir = Path(__file__).resolve().parent
_root_dir = _scripts_dir.parent
_src_dir = _root_dir / "src"

if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
