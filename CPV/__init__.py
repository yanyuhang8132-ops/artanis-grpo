from pkgutil import extend_path
from pathlib import Path
import sys

# turn CPV into a namespace-like package
__path__ = extend_path(__path__, __name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_PARENT = REPO_ROOT / "src" / "verifier" / "legacy"   # contains folder "CPV"

# ensure Python can find legacy CPV modules
if LEGACY_PARENT.exists() and str(LEGACY_PARENT) not in sys.path:
    sys.path.insert(0, str(LEGACY_PARENT))

LEGACY_CPV = LEGACY_PARENT / "CPV"
if LEGACY_CPV.exists():
    # allow importing submodules from legacy/CPV as CPV.xxx
    if str(LEGACY_CPV) not in __path__:
        __path__.append(str(LEGACY_CPV))
