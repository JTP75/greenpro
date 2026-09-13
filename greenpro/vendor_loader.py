"""Loads model wrapper classes from vendor/opencv_zoo without touching
sys.path or the package system.

vendor/opencv_zoo is a sparse checkout of https://github.com/opencv/opencv_zoo
(Apache 2.0 -- see vendor/opencv_zoo/LICENSE and each model's own LICENSE
file), trimmed to the three model directories this project uses. It's vendored
rather than reimplemented because at least one of them
(person_detection_mediapipe/mp_persondet.py) embeds a large literal SSD anchor
array that must match the model's training exactly -- transcribing or
regenerating that by hand is a correctness risk with no way to self-check on
this machine, whereas using their file verbatim inherits their own tested
behavior directly.

Each vendored file is a self-contained module (only `numpy`/`cv2` imports, no
relative imports), so loading it by file path with a private module name
avoids any risk of colliding with this project's own module names (notably:
this repo has its own top-level models/ directory for downloaded weights,
distinct from vendor/opencv_zoo/models/).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

VENDOR_MODELS_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "opencv_zoo" / "models"

_cache: dict[str, ModuleType] = {}


def load_module(relative_path: str) -> ModuleType:
    """relative_path is e.g. 'object_detection_nanodet/nanodet.py'."""
    if relative_path in _cache:
        return _cache[relative_path]

    file_path = VENDOR_MODELS_ROOT / relative_path
    if not file_path.exists():
        raise FileNotFoundError(
            f"vendored model source not found: {file_path}. "
            "Expected a sparse checkout of opencv/opencv_zoo under vendor/opencv_zoo -- "
            "see docs/models.md."
        )

    module_name = "_greenpro_vendor_" + relative_path.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load vendored module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _cache[relative_path] = module
    return module


def model_weights_path(relative_path: str) -> str:
    """Resolve a vendored .onnx weight file path, e.g.
    'object_detection_nanodet/object_detection_nanodet_2022nov.onnx'."""
    path = VENDOR_MODELS_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"vendored model weights not found: {path}")
    return str(path)
