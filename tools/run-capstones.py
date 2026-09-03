#!/usr/bin/env python3
"""Extract and execute the python block of every chapter capstone.

Each book/<chapter>/10-putting-it-together.md (and its book-zh/ translation) ships a zero-dependency,
deterministic, stdlib-only runnable. This gate re-runs all of them so an edit
can never silently break a code block readers are told they can execute.

Usage: python3 tools/run-capstones.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CAPSTONES = sorted(
    p for root in ("book", "book-zh") for p in Path(root).glob("*/10-putting-it-together.md")
)
if not CAPSTONES:
    sys.exit("no capstone files found; run from the repo root")

failures = []
for md in CAPSTONES:
    m = re.search(r"```python\n(.*?)```", md.read_text(), re.S)
    if not m:
        failures.append(f"{md}: no python code block")
        continue
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(m.group(1))
        path = f.name
    r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        failures.append(f"{md}: exit {r.returncode}\n{r.stderr.strip()[:400]}")
    else:
        print(f"ok  {md.parent.parent.name}/{md.parent.name}")

if failures:
    print(f"\n{len(failures)} capstone(s) FAILED:", file=sys.stderr)
    for f in failures:
        print("  " + f, file=sys.stderr)
    sys.exit(1)
print(f"\nall {len(CAPSTONES)} capstone runnables passed")
