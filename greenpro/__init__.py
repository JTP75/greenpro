"""greenpro: camera capture and processing pipeline for the MIT Green Building
17x9 RGB display.

See the top-level README.md for project context and docs/ for current-state
notes. This package covers capture -> segmentation -> geometry/downscale ->
local preview server -> network delivery to the building's simulator server
(see greenpro/sink.py and docs/simulator.md for the wire protocol).
"""

__version__ = "0.1.0"
