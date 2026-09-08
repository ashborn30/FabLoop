# Environment setup and preflight

This directory records dependency installation and checks before running the
EfficientAD/SAM3 pipeline. No training, calibration, evaluation, benchmark, or
model inference is performed by these checks.

## Verified result (2026-09-08)

- Installed 97 packages, plus pip (98 installed distributions).
- `python -m pip check`: **No broken requirements found**.
- Re-resolving the final `requirements.txt` with uv: **Would make no changes**.
- Preflight: **53 PASS, 20 BLOCKED_DATA, 4 INFO**, with `packages_valid: true`,
  `environment_valid: true`, and `pipeline_ready: false`.
- All 10 modules in `src`, the local editable SAM3 package, GUI dependencies,
  and the checked EfficientAD/SAM3 signatures passed.
- `python -B src/anomaly.py --help`: exit 0; no stage was started.
- PyTorch **2.11.0+cu128** recognized CUDA **12.8** and the
  **NVIDIA GeForce RTX 3050 Ti Laptop GPU**.
- Separate synthetic checks passed: PNG encode/decode, TIFF deflate roundtrip,
  small CUDA tensor arithmetic, torchvision CUDA NMS, and SAM3 Triton connected
  components / EDT on a 32x32 mask. EDT agreed with SciPy's distance transform.

The repeated `BLOCKED_DATA` entries refer to missing dataset roots and their
required subdirectories/files, rather than 20 independent package failures.
The four `INFO` entries identify Imagenette's `DOWNLOAD_ON_TRAIN` behavior.
Upstream FrEIA/timm/Anomalib emitted nonfatal syntax/deprecation warnings.

Evidence: `environment_check.json`, `dependency_smoke.json`,
`installed-packages.txt`, `installation_check.json`, and `third_party_audit.md`.

## Environment

- Use `.venv` at the project root, with Python 3.12.13.
- The current folder contains only `.venv`; the old `.venv-py313-backup` and
  `venv` directories are no longer present. The original requirements backup remains.
- `.vscode/settings.json` sets the default interpreter to
  `.venv/Scripts/python.exe`.
- If VS Code has already saved a different interpreter for this workspace,
  select `.venv/Scripts/python.exe` with **Python: Select Interpreter**.

From a PowerShell terminal in the project root:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pip check
python scripts\check_environment.py
```

Activation applies to that terminal. The explicit interpreter works without
activation or any execution-policy changes:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_environment.py
```

The preflight command writes `docs/preflight/environment_check.json`. A missing
configured dataset makes the pipeline unready even when the environment imports
successfully; inspect the separate environment and pipeline statuses in its
output. Do not use `src/anomaly.py run` as an environment check.

## Why requirements changed

`requirements.original.txt` is an unchanged copy of the initial file. That file
could not resolve on Windows: `nvidia-cufile-cu12==1.13.1.3` has no Windows wheel.
It also conflicted with the copied SAM3 package's `numpy>=1.26,<2` declaration.

The current root `requirements.txt` keeps the original pins except where needed:

| Dependency | Adjustment | Reason |
| --- | --- | --- |
| Python / NumPy | Python 3.12 / NumPy 1.26.4 | Install supported NumPy 1.x wheels without altering SAM3 metadata |
| OpenCV headless | 4.11.0.86 | Original 5.0.0.93 requires NumPy 2 |
| SciPy | 1.17.1 | Original 1.18.0 requires NumPy 2 |
| imagecodecs | 2025.3.30 | Original 2026.6.26 requires NumPy >=2.1 |
| tifffile | 2025.5.10 | Match the compatible NumPy 1.x / imagecodecs stack |
| NVIDIA `nvidia-*` packages | Linux platform markers | Windows PyTorch CUDA wheels supply their runtime libraries |
| Triton | Linux: 3.6.0; Windows: triton-windows 3.6.0.post26 | Use the Windows build matching PyTorch 2.11 |
| SAM3 | Editable local install | Register the source at `third-party/SAM3` for normal imports |
| PyTorch index | Add official CUDA 12.8 index | Resolve the existing `+cu128` pins |

The resolver/install used uv with `--python .venv/Scripts/python.exe`,
`--index-strategy unsafe-best-match`, and `--link-mode copy`. The index strategy
allows the explicitly pinned CUDA wheels and PyPI dependencies to resolve across
the two configured indexes. Installation logs are in `install.log`.

Primary references:

- [PyTorch CUDA wheel installation](https://pytorch.org/get-started/previous-versions/)
- [Triton Windows compatibility](https://github.com/triton-lang/triton-windows)
- [SAM3 installation and prerequisites](https://github.com/facebookresearch/sam3#installation)
- [Tifffile 2025.5.10](https://pypi.org/project/tifffile/2025.5.10/)

## Scope and remaining inputs

The YAML files currently reference `data/visa`, `data/mvtec_ad`, and
`data/mvtec_loco`. Those directories are absent; the available `PCBA_4Light` and
`PCBA_4Light_edited` directories do not implement the configured datasets.
`data/imagenette` is also absent and the training code is designed to download it
when needed. Dataset paths were not redirected to unrelated data.

The SAM3 checkpoint and tokenizer are present. A valid checkpoint ZIP structure
does not verify every tensor or guarantee successful model loading. The RTX 3050
Ti Laptop GPU has 4 GiB VRAM; full SAM3/EfficientAD memory use and inference quality
remain untested. See `third_party_audit.md` for source-copy and API findings.
