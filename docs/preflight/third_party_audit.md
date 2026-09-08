# Third-party integration audit

Date: 2026-09-08. Workspace: `D:\Visual studio code\FabLoop_VJSS26`.

## Scope and conclusion

This is a preparation audit of the copied third-party sources against `src` and
`configs`. The audit used source inspection, Python standard-library AST parsing,
CSV checks, gzip decompression, Git status, and checkpoint ZIP-directory reads.
It did not train a model, run inference, execute a dataset stage, or modify
`src`, `configs`, or third-party sources. No `AGENTS.md` was found in the inspected
project tree.

The SAM3 copy is structurally complete for the project's imports, and the wrapper
matches the copied API. The VisA split CSV is sufficient for the project's use
of `spot-diff`. This does not establish successful full-model inference or
numerical correctness. The configured datasets are missing, so the current
configs are not ready for an end-to-end run.

## SAM3 source and assets

- Repository: [third-party/SAM3](../../third-party/SAM3).
- Checked Git commit: `96914d2425f90a64f45ca977c2b5165418099543`.
- `git status --short` returned no changes at audit time. This verifies the
  working copy against its local Git checkout; it is not remote provenance
  verification.
- Parsed all 162 Python files found under `src` and `third-party/SAM3/sam3` with
  `ast.parse`: zero syntax errors. Absolute `sam3.*` import targets referenced by
  these files all resolve to a copied Python module or package directory.
  These checks do not execute imports or validate every optional dependency.
- The bundled vocabulary
  `third-party/SAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz` decompresses cleanly
  and contains 262,145 lines.
- `third-party/SAM3/checkpoints/sam3.pt` exists and is 3,450,062,241 bytes.
  Its ZIP directory opens successfully and contains 1,471 entries, including
  `sam3/data.pkl`. `checkpoints/config.json` also exists.
- Checkpoint inspection did not unpickle the checkpoint, read all tensor data,
  validate every ZIP member's CRC, compare a trusted upstream hash, or verify
  loaded state-dict keys. A readable ZIP directory alone is not proof that all
  weights are intact or compatible.
- Full SAM3 construction and inference remain untested on the machine's 4 GB
  GPU. No claim is made that the complete model and inference workload fit in
  that GPU's memory.

## Wrapper and configuration compatibility

[src/sam_refiner.py](../../src/sam_refiner.py), lines 106-126, imports
`Sam3Processor` and `build_sam3_image_model` and passes keyword arguments present
in [model_builder.py](../../third-party/SAM3/sam3/model_builder.py), lines 573-582.
The copied builder supports `enable_inst_interactivity=True`, which the wrapper
requires for `predict_inst`.

The wrapper's calls to `set_image` and `predict_inst` (lines 185-192 and 301-308)
match the copied processor and image model API. The copied
`SAM3InteractiveImagePredictor.predict` accepts the point and box prompts and
returns NumPy masks; the wrapper handles the single-box squeezed shape and the
multiple-box batch shape. This is source-level agreement, not a model execution
test.

The configured `processor_resolution: 1008` in
[configs/efficientad.yaml](../../configs/efficientad.yaml), line 58, agrees with
the copied predictor's hardcoded feature sizes `(288, 288)`, `(144, 144)`, and
`(72, 72)` in
[sam1_task_predictor.py](../../third-party/SAM3/sam3/model/sam1_task_predictor.py),
lines 61-65. Arbitrary changes to this resolution have not been validated.

The VisA and MVTec config schemas match the routing in
[src/data.py](../../src/data.py): legacy `data.root` / top-level `category` for
VisA, and `dataset.root` / `dataset.category` for MVTec. VisA and MVTecAD use the
configured train-normal calibration holdout; MVTecLoco uses official validation.

### Environment integration requirements

- The copied package needs installation into the selected virtual environment;
  merely keeping `sam3.egg-info` in the copied directory does not register the
  package in a different environment. Editable installation is the appropriate
  local-source registration. The root setup task handles installation and
  runtime import checks separately from this audit.
- SAM3 declares `numpy>=1.26,<2` in
  [pyproject.toml](../../third-party/SAM3/pyproject.toml), line 29. The original
  root requirements only specified `numpy>=1.24`; installation must satisfy
  SAM3's tighter constraint.
- Triton is imported eagerly even for the image wrapper:
  `sam3/__init__.py:5` -> `model_builder.py:40` ->
  `model/sam1_task_predictor.py:16` -> `model/sam3_tracker_base.py:10` ->
  `model/sam3_tracker_utils.py:9` -> `model/edt.py:8-9`.
  An appropriate Windows-compatible Triton installation is therefore needed
  for the unchanged copied source. The CUDA connected-components fallback also
  uses Triton when `cc_torch` is unavailable; the interactive predictor enables
  hole filling by default (`max_hole_area=256`).

### Latent indexed-device issue

All current configs use `device: cuda`, which agrees with the copied builder.
However, the CLI can accept `--device cuda:0` and the wrapper passes that string
to SAM3. In
[model_builder.py](../../third-party/SAM3/sam3/model_builder.py), lines 564-570,
`_setup_device_and_mode` only moves the model when `device == "cuda"`; an indexed
string such as `cuda:0` skips this move while the processor uses the requested
GPU. That can cause a device mismatch. A future bounded wrapper fix is an
explicit `model.to(device)` after building, with an appropriate execution test.
This audit leaves runtime code unchanged and does not treat the current default
`cuda` configuration as affected by that specific issue.

## spot-diff split CSV

The copied [third-party/spot-diff](../../third-party/spot-diff) contains the
`split_csv/1cls.csv` asset, not a full Python repository. This is sufficient for
the current code: `src` reads the CSV and does not import `spot-diff` Python
modules.

- File: [1cls.csv](../../third-party/spot-diff/split_csv/1cls.csv).
- SHA-256:
  `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995`.
- Columns are exactly `object,split,label,image,mask`, matching
  `src/data.py:23,117`.
- 10,821 data rows; zero duplicate image paths; zero anomaly rows with an empty
  mask path.
- All four PCB categories used by `src/data.py` are present:

| Category | Official train normal | Test normal | Test anomaly |
| --- | ---: | ---: | ---: |
| pcb1 | 904 | 100 | 100 |
| pcb2 | 901 | 100 | 100 |
| pcb3 | 905 | 101 | 100 |
| pcb4 | 904 | 101 | 100 |

These checks validate the local CSV structure and contents. They do not verify
image/mask file contents, because the configured VisA root is absent, and do not
compare the CSV against a fetched upstream reference.

## Missing configured data

At audit time, `data` contained only `PCBA_4Light` and `PCBA_4Light_edited`.

| Missing path | Referencing configuration | Consequence |
| --- | --- | --- |
| `data/visa` | `configs/efficientad.yaml:7` | VisA split construction cannot proceed. |
| `data/mvtec_loco` | `configs/pushpins.yaml:3`, `configs/splicing_connectors.yaml:3` | LOCO split construction cannot proceed. |
| `data/mvtec_ad` | `configs/transistor.yaml:3` | MVTecAD split construction cannot proceed. |
| `data/imagenette` | All four configs | The project reports this as `DOWNLOAD_ON_TRAIN`; it is not prepared offline. |

The dataset-root existence checks are in `src/data.py:262-263` and `333`.
`PCBA_4Light` is not a supported dataset route in these four configs; it should
not be silently substituted for VisA or MVTec. Obtaining the intended datasets
or adding a deliberate new dataset configuration is separate work before a full
run.
