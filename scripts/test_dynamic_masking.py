#!/usr/bin/env python3
"""Deprecated shim: use scripts/test_mlm_masking.py.

The original version of this script asserted only that two consecutive masking
calls produced different tensors. That property holds for a 100/0/0
implementation, which is how the 80/10/10 split was lost while the docstring
still claimed it was there (docs/AUDIT-2026-07.md item 1).

scripts/test_mlm_masking.py asserts the composition of the selected positions
(~80% [MASK] / 10% random / 10% unchanged), the special-token guard, label
correctness, and the selection rate. This shim delegates to it so any old
`python scripts/test_dynamic_masking.py` invocation still tests the right thing.

Run:  python scripts/test_mlm_masking.py
"""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    print(__doc__)
    print("Delegating to scripts/test_mlm_masking.py\n")
    runpy.run_path(str(Path(__file__).with_name("test_mlm_masking.py")), run_name="__main__")
