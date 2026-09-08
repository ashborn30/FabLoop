"""Launch or list the lightweight PCBA Pseudo-3D Viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from fabloop.photometric_stereo.normal_relight import discover_relight_cases


def default_results_root() -> Path | None:
    candidates = (
        WORKSPACE / "outputs" / "pcba_photometric_20260908",
        WORKSPACE.parent.parent / "outputs" / "pcba_photometric_20260908",
        Path.cwd() / "outputs" / "pcba_photometric_20260908",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--max-render-edge", type=int, default=1400)
    parser.add_argument("--list-only", action="store_true", help="Print discovered PCBA folders and exit.")
    args = parser.parse_args(argv)

    results_root = args.results_root or default_results_root()
    if results_root is None:
        print("No results root found. Pass --results-root PATH.", file=sys.stderr)
        return 1

    cases = discover_relight_cases(results_root)
    for case in cases:
        print(f"PCBA folder: {case.name} | normal={case.normal_path} | albedo={case.albedo_path}")
    if args.list_only:
        print(f"Found {len(cases)} PCBA folders.")
        return 0 if cases else 1

    from PySide6 import QtWidgets
    from fabloop.photometric_stereo.pseudo3d_viewer import Pseudo3DViewerWidget

    app = QtWidgets.QApplication(sys.argv[:1])
    widget = Pseudo3DViewerWidget(results_root, max_render_edge=args.max_render_edge)
    widget.resize(1280, 820)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
