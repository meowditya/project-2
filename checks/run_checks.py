"""Run every check (plain Python, no pytest needed).

    python checks/run_checks.py            # all checks
    python checks/run_checks.py exact      # only files whose name contains a word

Each check runs with Python warnings turned into errors. Exit status is
non-zero if anything fails.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import time
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(argv: list[str]) -> int:
    names = sorted(p.stem for p in HERE.glob("test_*.py"))
    if argv:
        names = [n for n in names if any(a in n for a in argv)]
    passed = failed = 0
    t0 = time.perf_counter()
    for name in names:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                mod = importlib.import_module(name)
        except Exception:
            failed += 1
            print(f"FAIL  {name} (could not import: module-level setup raised)")
            traceback.print_exc(limit=3)
            continue
        tests = sorted((f for n, f in inspect.getmembers(mod, inspect.isfunction)
                        if n.startswith("test_") and f.__module__ == mod.__name__),
                       key=lambda f: f.__code__.co_firstlineno)
        for fn in tests:
            label = f"{name}::{fn.__name__}"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    fn()
            except Exception:
                failed += 1
                print(f"FAIL  {label}")
                traceback.print_exc(limit=3)
            else:
                passed += 1
                print(f"ok    {label}")
    print(f"\n{passed} passed, {failed} failed in {time.perf_counter() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
