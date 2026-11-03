#!/usr/bin/env python3
"""Keep Challenge.lean's definitions identical to MaxelerPermanent/Defs.lean.

Challenge.lean may only import Mathlib, so it repeats the definitions of
MaxelerPermanent/Defs.lean verbatim; `lake comparator` then checks that the
statements in Solution.lean (which use the library's copy) are the same.

  python3 scripts/sync-challenge.py          # regenerate Challenge.lean
  python3 scripts/sync-challenge.py --check  # fail if Challenge.lean is stale
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "-- BEGIN DEFINITIONS (generated from MaxelerPermanent/Defs.lean)"
END = "-- END DEFINITIONS"


def definitions() -> str:
    text = (ROOT / "MaxelerPermanent/Defs.lean").read_text(encoding="utf-8")
    start = text.index("namespace MaxelerPermanent")
    stop = text.rindex("end MaxelerPermanent")
    return text[start:stop].rstrip() + "\n"


def render(challenge: str) -> str:
    head, rest = challenge.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n\n{definitions()}\nend MaxelerPermanent\n\n{END}{tail}"


def main() -> int:
    path = ROOT / "Challenge.lean"
    current = path.read_text(encoding="utf-8")
    wanted = render(current)
    if "--check" in sys.argv[1:]:
        if current != wanted:
            print("Challenge.lean is out of sync with MaxelerPermanent/Defs.lean; "
                  "run python3 scripts/sync-challenge.py")
            return 1
        return 0
    path.write_text(wanted, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
