"""Put the repo root on sys.path for pytest collection.

Each test file also carries its own shim so `python tests/<file>.py` works
standalone (that is how they have always been run — there is no CI). This file
only covers the `pytest tests/` entry point, where pytest imports the modules
rather than executing them as scripts.

Note that only test_metrics.py is written in pytest style (15 `test_*`
functions). The others are `__main__` scripts that print PASS/FAIL and are meant
to be run directly:

    python tests/test_patch_cache.py
    python tests/smoke_test.py                 # needs torchvision for the
    python tests/smoke_test_infer.py           #   perceptual scenarios
    python tests/smoke_test_organ_focus.py
    python tests/smoke_test_organ_weights.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
