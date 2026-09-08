"""Export normals and relative height from explicitly calibrated PCBA captures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from fabloop.photometric_stereo.pcba_process import process_capture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rps-root", type=Path, default=WORKSPACE / "third-party/RobustPhotometricStereo")
    parser.add_argument("--solver", choices=("l2", "l1", "l1-multicore"), default="l2")
    args = parser.parse_args(argv)
    try:
        report = process_capture(args.capture_config, args.output_dir, args.rps_root, args.solver)
    except (OSError, ValueError, ImportError) as error:
        print(f"PCBA photometric processing failed: {error}", file=sys.stderr)
        return 1
    print(f"{report['status']}: normals {report['image_shape']}; relative height in pixel units.")
    print(f"Report: {args.output_dir.resolve() / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
