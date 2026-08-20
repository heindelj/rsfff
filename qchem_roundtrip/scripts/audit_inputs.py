#!/usr/bin/env python3
"""Report which qchem_roundtrip inputs are runnable versus already completed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def lock_claims(calc_dir: Path) -> list[dict[str, object]]:
    claims = []
    for claim_path in sorted((calc_dir / "state" / "locks").glob("*.lock/claim.json")):
        try:
            claim = json.loads(claim_path.read_text())
        except (OSError, json.JSONDecodeError):
            claim = {}
        claim["lock"] = str(claim_path.parent)
        claims.append(claim)
    return claims


def format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-active", action="store_true", help="Print active lock claim details.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    for inputs_dir in sorted(path for path in root.rglob("inputs") if path.is_dir()):
        calc_dir = inputs_dir.parent
        inputs = {path.stem for path in (calc_dir / "inputs").glob("*.in")}
        outputs = {path.stem for path in (calc_dir / "outputs").glob("*.out")}
        done = {path.stem for path in (calc_dir / "state" / "done").glob("*.json")}
        failed = {path.stem for path in (calc_dir / "state" / "failed").glob("*.json")}
        claims = lock_claims(calc_dir)
        completed = outputs | done | failed
        runnable = inputs - completed
        label = str(calc_dir.relative_to(root))
        print(
            f"{label:24s} "
            f"inputs={len(inputs):6d} outputs={len(outputs):6d} "
            f"done={len(done):6d} failed={len(failed):6d} active={len(claims):6d} "
            f"runnable={len(runnable):6d} skipped_by_worker={len(inputs & completed):6d}"
        )
        if args.show_active:
            now = time.time()
            for claim in claims:
                input_name = Path(str(claim.get("input", ""))).name
                host = claim.get("hostname", "?")
                pid = claim.get("pid", "?")
                claimed_at = claim.get("claimed_at")
                age = format_age(now - float(claimed_at)) if isinstance(claimed_at, (int, float)) else "?"
                print(f"  active input={input_name} host={host} pid={pid} age={age}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
