"""Audit grouped PCBA captures before calibrated processing and training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from fabloop.photometric_stereo.pcba_prepare import prepare_inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edited-root", type=Path, default=WORKSPACE / "data/PCBA_4Light_edited")
    parser.add_argument("--raw-root", type=Path, default=WORKSPACE / "data/PCBA_4Light")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "data/processed/pcba_4light")
    parser.add_argument("--summary", type=Path, default=WORKSPACE / "docs/preflight/pcba_4light_readiness.json")
    parser.add_argument("--labels", type=Path, help="Manual labels JSON; defaults to output-dir/labels.json")
    parser.add_argument("--require-ready", action="store_true", help="Return 1 when processing/training prerequisites are unresolved")
    args = parser.parse_args(argv)
    try:
        result = prepare_inventory(args.edited_root, args.raw_root, args.output_dir, args.summary, args.labels)
    except (OSError, ValueError) as error:
        print(f"Inventory failed: {error}", file=sys.stderr)
        return 1
    print(f"Inventory completed: {result['counts']}")
    print(f"Readiness: {result['status']}; ready_for_training={result['ready_for_training']}")
    print(f"Inventory: {result['inventory_path']}\nLabels: {result['labels_path']}\nSummary: {args.summary.resolve()}")
    return int(args.require_ready and not result["ready_for_training"])


if __name__ == "__main__":
    raise SystemExit(main())
