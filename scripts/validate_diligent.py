"""Run the migrated DiLiGenT adapter with the project interpreter, from any cwd."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabloop.photometric_stereo.diligent_validate import main


if __name__ == "__main__":
    raise SystemExit(main())
