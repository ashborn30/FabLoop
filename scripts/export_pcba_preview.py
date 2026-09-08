"""Export qualitative normal/height previews from uncalibrated PCBA four-light captures.

The --nominal-lights flag explicitly selects assumed lighting. These exports are
visual experiments, not calibrated geometry or a prepared training dataset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from fabloop.photometric_stereo.pcba_preview import export_preview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=WORKSPACE / "data/PCBA_4Light_edited")
    parser.add_argument("--output-root", type=Path, default=WORKSPACE / "outputs/pcba_photometric_preview")
    parser.add_argument("--boards", nargs="+", help="Board folders, e.g. PCB1 PCB2; default: every PCB<number> folder")
    parser.add_argument("--max-edge", type=int, default=1024, help="Largest image edge in pixels; aspect ratio preserved")
    parser.add_argument("--nominal-lights", action="store_true", required=True,
                        help="Use explicitly assumed light directions/intensities for qualitative previews")
    args = parser.parse_args(argv)
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    try:
        if not input_root.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_root}")
        if not 128 <= args.max_edge <= 4096:
            raise ValueError("--max-edge must be between 128 and 4096")
        if input_root == output_root or input_root in output_root.parents or output_root in input_root.parents:
            raise ValueError("Output must be separate from the source capture tree")
        if (output_root / "summary.json").exists():
            raise FileExistsError("Output already contains a summary; choose a new --output-root to preserve the previous run")
        names = args.boards or sorted(
            (path.name for path in input_root.iterdir() if path.is_dir() and re.fullmatch(r"PCB\d+", path.name)),
            key=lambda name: int(name[3:]),
        )
        if not names or len(set(names)) != len(names):
            raise ValueError("Select at least one distinct board folder")
        for name in names:
            if not re.fullmatch(r"PCB\d+", name):
                raise ValueError(f"Invalid board folder: {name!r}")
            source = (input_root / name).resolve()
            if source.parent != input_root or not source.is_dir():
                raise ValueError(f"Board must be an existing direct child of input root: {name}")
            destination = output_root / name
            if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
                raise FileExistsError(f"Board output is not empty; choose a new output root: {destination}")
        output_root.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"Preview setup failed: {error}", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    print("QUALITATIVE PREVIEW: nominal lights, unverified registration, no training readiness.", flush=True)
    for index, name in enumerate(names, start=1):
        print(f"[{index}/{len(names)}] {name}", flush=True)
        try:
            report = export_preview(input_root / name, output_root / name, max_edge=args.max_edge)
            records.append({"board": name, "status": report["status"],
                            "report": str(output_root / name / "report.json")})
            print(f"  Exported: {output_root / name}", flush=True)
        except Exception as error:
            # Keep per-board failure evidence and continue independent captures.
            records.append({"board": name, "status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
            print(f"  FAILED: {error}", file=sys.stderr, flush=True)
    failed = sum(record["status"] == "FAILED" for record in records)
    summary = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL_FAILURE" if failed else "QUALITATIVE_PREVIEW",
        "input_root": str(input_root), "output_root": str(output_root), "max_edge": args.max_edge,
        "calibrated": False, "registration_verified": False, "training_ready": False,
        "normal_ground_truth_available": False,
        "nominal_lights_explicitly_selected": True,
        "counts": {"requested": len(names), "exported": len(names) - failed, "failed": failed},
        "boards": records,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(names) - failed}/{len(names)}; failed {failed}. Summary: {summary_path}", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
