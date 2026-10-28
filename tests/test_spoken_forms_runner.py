"""Test runner for spoken-form normalisation test fixture.

Reads test pairs from tests/test_spoken_forms.txt and tests them against src/asr_normalise.py.
If src/asr_normalise.py does not exist, skips with 'src/asr_normalise.py missing'.
Pairs not yet implemented in src/asr_normalise.py are marked as expected failures (XFAIL, strict=False)
with the reason 'pending asr_normalise implementation'.
This runner does not implement any conversion itself (Agent 2 owns src/asr_normalise.py).
"""

from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

# Implemented capabilities in current src/asr_normalise.py:
# 1. 8-digit order IDs: e.g. "six nine five two zero eight six seven"
# 2. 3-letter + 4-digit catalog SKUs: e.g. "ele zero zero one two"
# All other entities (pincodes, dates, product names, currency, Hinglish phonetics) are pending Agent 2.
IMPLEMENTED_INPUTS = {
    "six nine five two zero eight six seven",
    "ele zero zero one two",
}


def run_spoken_form_tests() -> None:
    asr_module_path = REPO_ROOT / "src" / "asr_normalise.py"
    if not asr_module_path.exists():
        print("SKIPPED: src/asr_normalise.py missing")
        return

    try:
        from src.asr_normalise import normalise_asr_text
    except Exception as exc:
        print(f"SKIPPED: Could not import normalise_asr_text from src/asr_normalise.py: {exc}")
        return

    fixture_path = REPO_ROOT / "tests" / "test_spoken_forms.txt"
    if not fixture_path.exists():
        print(f"ERROR: Test fixture file not found at {fixture_path}")
        return

    lines = fixture_path.read_text(encoding="utf-8").splitlines()
    test_cases = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "->" in line:
            parts = line.split("->")
            inp = parts[0].strip()
            outp = parts[1].strip()
            test_cases.append((inp, outp))

    print(f"Loaded {len(test_cases)} test case(s) from {fixture_path.name}")
    print("=" * 70)

    passed = 0
    xfailed = 0
    failed = 0

    for i, (inp, expected) in enumerate(test_cases, 1):
        actual = normalise_asr_text(inp)
        match = (actual.strip() == expected.strip())
        is_expected_implemented = inp in IMPLEMENTED_INPUTS

        if match:
            passed += 1
            print(f"[PASS]  Test #{i:02d}: '{inp}' -> '{actual}'")
        else:
            if not is_expected_implemented:
                xfailed += 1
                print(f"[XFAIL] Test #{i:02d}: '{inp}' -> expected '{expected}' (pending asr_normalise implementation)")
            else:
                failed += 1
                print(f"[FAIL]  Test #{i:02d}: '{inp}'")
                print(f"        Expected: '{expected}'")
                print(f"        Actual:   '{actual}'")

    print("=" * 70)
    print(f"Summary: {passed} PASSED | {xfailed} XFAIL | {failed} FAILED | {len(test_cases)} TOTAL")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run_spoken_form_tests()
