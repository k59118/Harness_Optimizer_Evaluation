#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __name__ != "__main__":
    from . import run_shor_inner as _inner

    sys.modules[__name__] = _inner
else:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from shor.run_shor_inner import main

    main()
