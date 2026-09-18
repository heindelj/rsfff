"""Deprecated: the driver moved to ``workflows.py``.

``workflows.py`` is where the loop is wired up now -- it has the Q-Chem label stage, the task
spec, and hooks for submitting and syncing the cluster jobs. This forwards so an old command
line keeps working; use ``python workflows.py water --root ...`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflows import main  # noqa: E402

if __name__ == "__main__":
    print("run_water.py is deprecated; use: python workflows.py water --root ...\n",
          file=sys.stderr)
    raise SystemExit(main(["water", *sys.argv[1:]]))
