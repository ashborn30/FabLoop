# EfficientAD source snapshot and Slim-0.5 candidate

`torch_model.py` is a byte-for-byte copy of the Anomalib **2.6.0** implementation
installed in this project's `.venv`. Its SHA-256 matches the installed wheel's
`RECORD`; see `SOURCE.json`. The original copyright and Apache-2.0 SPDX headers
are retained, with the upstream license in `LICENSE`.

The frozen file remains the baseline. `slim_model.py` now implements the first
**Slim-0.5 candidate**, with only the Student and Autoencoder internal channel
widths reduced. This is a candidate for experiments, not the final architecture.
The active pipeline still constructs the official Anomalib model in
`src/model.py`; training configuration and checkpoint routing have not switched.

The candidate preserves the feature contract:

| Network | Fixed output | Candidate channel sequence |
| --- | --- | --- |
| Teacher-S | 384 | Original `3 -> 128 -> 256 -> 256 -> 384` |
| Slim Student | 768 = 384 ST + 384 SA | `3 -> 64 -> 128 -> 128 -> 768` |
| Slim AE encoder | 32 (internal bottleneck) | `3 -> 16 -> 16 -> 32 -> 32 -> 32 -> 32` |
| Slim AE decoder | 384 | Seven hidden convolutions at 32 channels, then `32 -> 384` |

Layer count, kernel size, stride, pooling, padding, spatial output, activation,
interpolation and dropout are retained. Forward methods, distance calculations,
hard-feature selection, ImageNet penalty, augmentations, map construction and
calibration logic are inherited from the frozen source. The Teacher keeps its
original architecture, has gradients disabled, and stays in evaluation mode
when the candidate enters training mode.

From the repository root, with `.venv` active:

```python
from src.models.efficientad_slim import (
    SlimAutoEncoder,
    SlimEfficientAdModel,
    SlimStudent,
)

student = SlimStudent()               # fixed output: 768 channels
autoencoder = SlimAutoEncoder()       # fixed output: 384 channels
model = SlimEfficientAdModel(padding=False, pad_maps=True)
model.eval()
```

These constructors initialize weights. They do not load a pretrained Teacher or
a trained checkpoint. Loading the verified pretrained Teacher and retaining the
existing Teacher checksum gates are required when integrating training.

## Mandatory shape gate

```python
from src.models.efficientad_slim.shape_check import check_feature_shapes

shape_report = check_feature_shapes(model, image_size=(256, 256))
```

The gate checks Teacher, Student, both 384-channel Student slices and AE with
exact shape equality before evaluating `T - S_T`, `T - A`, and `A - S_A`.
It rejects broadcasting, missing channels and non-finite features/differences.
Model state, existing gradients, mixed train/eval flags and PyTorch RNG are
preserved. Raw feature subtraction checks do not replace the original loss.

The standalone command `python scripts/check_efficientad_shapes.py` passed on
CPU and CUDA for `[1,3,256,256]`: Teacher/AE/heads `[1,384,56,56]`, Student
`[1,768,56,56]` with padding disabled. `src/trainer.py` now runs the same gate
before Teacher/Imagenette preparation and optimizer creation. The existing
factory still selects baseline; Slim selection remains a later integration step.

Run all 19 architecture, shape-gate and compression-gate regressions with
`python -m unittest discover -s tests -p "test_*.py" -v`.

Run the synthetic architecture checks and comparison profiler:

```powershell
python -m unittest discover -s tests -p test_efficientad_slim.py -v
python scripts/profile_efficientad_slim.py
```

The seven tests pass: geometry and inherited methods, square/non-square shapes
with both padding modes, finite loss/backward gradients for both Student output
groups and AE with no Teacher gradients, unchanged seeded Teacher, raw and
normalized map parity with the original core logic, strict state-dict roundtrip,
source hash, and invalid output/checkpoint rejection. Backward checks use
synthetic tensors and no optimizer step.
See [validation results](../../../docs/preflight/efficientad_slim_05_validation.json).

At CPU float32 input `[1, 3, 256, 256]`, eval mode, `padding=False`:

| Scope | Candidate params | fvcore-supported FLOPs | Params reduction | FLOPs reduction |
| --- | ---: | ---: | ---: | ---: |
| Student | 1,855,552 | 7,625,419,776 | 56.52% | 62.33% |
| AE | 330,240 | 781,088,256 | 69.88% | 67.44% |
| Teacher + Student + AE | 4,879,936 | 23,717,411,328 | 39.44% | 37.51% |

The Student passes the architecture targets of at least 40% fewer parameters
and 30% fewer supported FLOPs. `scripts/profile_efficientad_slim.py` saves both
PASS and FAIL reports, returning exit code 0 only if both Student targets pass,
or 1 if either fails. Thresholds use unrounded reductions. A failure requires
architecture changes before training; a pass permits the next integration step
and does not start training or switch the active baseline factory.
The core total covers the three feature networks;
it excludes 772 calibration parameter elements, anomaly-map postprocessing and
SAM3. fvcore counts one multiply-add as one FLOP and omits unsupported or
deliberately ignored operators. Full counting policy, source hashes and baseline
comparison are in
[efficientad_slim_05_profile.json](../../../docs/preflight/efficientad_slim_05_profile.json).
These measurements use random weights and synthetic input; they provide no
dataset quality, latency or Jetson result.

Future edits to copied files must carry a modification notice and preserve attribution.
Retain the original `SOURCE.json` provenance and record the slim implementation's
own version/hash separately. Source-file hashes do not replace the pretrained
Teacher weight checksum gates. Do not change `site-packages/anomalib`.

See the [readiness report](../../../docs/preflight/efficientad_slim_readiness.md)
for the remaining factory/config, checkpoint provenance, output routing and data
integration work before training.
