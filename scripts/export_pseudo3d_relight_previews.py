"""Export one relit pseudo-3D PNG per PCBA result folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from fabloop.photometric_stereo.normal_relight import (
    discover_relight_cases,
    light_direction,
    load_relight_inputs,
    render_relight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--azimuth", type=float, default=315.0)
    parser.add_argument("--elevation", type=float, default=45.0)
    parser.add_argument("--max-edge", type=int, default=1600)
    parser.add_argument("--boards", nargs="*", default=None)
    args = parser.parse_args(argv)

    selected = set(args.boards or [])
    cases = discover_relight_cases(args.results_root)
    if selected:
        cases = [case for case in cases if case.name in selected]
    if not cases:
        print("No PCBA relight cases found.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    light = light_direction(args.azimuth, args.elevation)
    for case in cases:
        inputs = load_relight_inputs(case, max_edge=args.max_edge)
        rgb = render_relight(inputs.normals, inputs.albedo, args.azimuth, args.elevation)
        out_path = args.output_dir / f"{case.name}_relight_az{args.azimuth:.0f}_el{args.elevation:.0f}.png"
        Image.fromarray(rgb, mode="RGB").save(out_path)
        print(f"Rendered {case.name}: {out_path}")
        records.append(
            {
                "board": case.name,
                "normal_path": str(case.normal_path),
                "albedo_path": str(case.albedo_path),
                "output": str(out_path),
                "azimuth_deg": args.azimuth,
                "elevation_deg": args.elevation,
                "light_direction": light.tolist(),
                "render_shape": list(rgb.shape),
            }
        )
    (args.output_dir / "summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(records)} PCBA folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
