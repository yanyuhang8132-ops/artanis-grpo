import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LEGACY_DIR = REPO_ROOT / "src" / "verifier" / "legacy" 

if LEGACY_DIR.exists() and str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))
