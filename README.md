# AlphaGenome batch variant effects from REF/ALT sequence pairs

This repo is a small command-line wrapper for running AlphaGenome-style sequence-to-function prediction on many REF/ALT sequence pairs.

It is designed for an environment with PyTorch, NumPy, pandas, and SciPy, and it does not import `huggingface_hub`, `transformers`, or other Hugging Face Python packages.

The CLI reads a CSV or TSV with sequence columns, runs predictions for each REF and ALT sequence, and writes two outputs:

1. `variant_effects.tsv`: per-track variant-effect summaries only.
2. `all_predictions.zip`: all raw model predictions for REF and ALT sequences as `.npy` arrays inside a zip archive, plus a manifest.

The repo includes a `mock` backend so we can test the I/O and output layout without model weights. To run real AlphaGenome predictions, install or vendor a PyTorch AlphaGenome implementation and place the model artifact under `artifacts/`.

## Repository layout

```text
alphagenome_variant_effects_repo/
  README.md
  requirements.txt
  pyproject.toml
  examples/input.tsv
  scripts/download_hf_without_python_hf.sh
  scripts/run_mock_example.sh
  src/ag_variant_effects/
    __main__.py
    archive.py
    backends.py
    cli.py
    io_utils.py
    predictions.py
    scoring.py
    sequence.py
```

## Input format

Input can be TSV or CSV. We choose the REF and ALT sequence column names at runtime.

Example `examples/input.tsv`:

```tsv
variant_id	ref_seq	alt_seq
var_1	ACGTACGTACGTACGTACGTACGTACGTACGT	ACGTACGTACGTTCGTACGTACGTACGTACGT
var_2	GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT	GGGGCCCCAAAATTTTGGGGCCCCAAAGTTTT
```

Sequences should be A/C/G/T. Ambiguous bases such as N are encoded as all-zero vectors by default. Use `--ambiguous uniform` or `--ambiguous raise` to change this.

## Install this wrapper

From the repo root, either install the wrapper without dependency resolution (useful when the environment already has PyTorch/NumPy/pandas/SciPy):

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

Or run without installing by setting `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m ag_variant_effects --help
```

The core wrapper dependencies are intentionally minimal:

```bash
python -m pip install -r requirements.txt
```

For real AlphaGenome inference, this wrapper expects a model backend. The included real backend uses `alphagenome_pytorch.AlphaGenome` if that package/module is available in your environment. The wrapper itself does not download from Hugging Face and does not use Hugging Face Python libraries.

## Obtain model artifacts

See `MODEL_ARTIFACTS.md` for detailed artifact paths.

### Recommended for your environment: PyTorch-converted artifact

Use a local PyTorch checkpoint supported by `alphagenome_pytorch.AlphaGenome.from_pretrained(...)`, for example:

```text
artifacts/model_all_folds.safetensors
# or
artifacts/alphagenome.pt
# or another .pth/.pt/.safetensors checkpoint accepted by your installed alphagenome_pytorch version
```

If your environment has no Hugging Face Python package but can access the Hugging Face website, you can download with a browser or with `curl` using a token. No Python Hugging Face dependency is required:

```bash
export HF_TOKEN='paste_your_token_here'
mkdir -p artifacts
bash scripts/download_hf_without_python_hf.sh model_all_folds.safetensors artifacts/model_all_folds.safetensors
```

This script uses only `curl`. It downloads from the public model repository path and passes your token in the HTTP Authorization header. You still need to follow the model terms/license flow for the artifact source.

If your environment cannot access Hugging Face at all, download the official JAX checkpoint from Kaggle in a separate environment, convert it to the PyTorch format using the conversion script from the PyTorch port, then copy the converted checkpoint into `artifacts/`.

### Official Google DeepMind artifacts

Google DeepMind's official `alphagenome_research` repo provides JAX model code and says pretrained weights are available from Kaggle or Hugging Face after accepting the non-commercial model terms. The official open-source model runner needs JAX/CUDA and recommends high-end hardware. This wrapper does not run the official JAX model directly because your target environment is PyTorch/TensorFlow-only.

Practical paths:

1. **Use the hosted AlphaGenome API** when you do not need local weights or massive offline inference.
2. **Use official Kaggle JAX weights** when you have a JAX-capable machine.
3. **Use PyTorch-converted weights** when your inference environment is PyTorch-only.

## Run a no-artifact smoke test

```bash
bash scripts/run_mock_example.sh
```

This writes:

```text
results/mock_variant_effects.tsv
results/mock_all_predictions.zip
```

The mock backend is only for checking input parsing, batching, scoring, and output writing.

## Run real AlphaGenome PyTorch inference

python -m ag_variant_effects \
  --backend alphagenome-pytorch \
  --weights artifacts/model_all_folds.safetensors \
  --input examples/input.tsv \
  --input-sep auto \
  --ref-col ref_seq \
  --alt-col alt_seq \
  --id-col variant_id \
  --organism human \
  --device cuda \
  --batch-size 1 \
  --effects-out results/variant_effects.tsv \
  --predictions-out results/all_predictions.zip
```

Use `--device cpu` for CPU-only testing, but real AlphaGenome inference is expected to be slow and memory-heavy on CPU.

## Output 1: variant effects only

`variant_effects.tsv` is a long table with one row per input sequence pair, output head, resolution, and track. The core columns are:

```text
input_id
row_index
head
resolution
track_index
track_name
ref_mean
alt_mean
delta_mean
ref_sum
alt_sum
delta_sum
log2fc_sum
ref_max
alt_max
delta_max
max_abs_delta
l2_delta
ref_shape
alt_shape
comparable_by_position
track_metadata_json
```

The effect summaries are generic because your input contains only REF/ALT sequences, not genomic coordinates, variant positions, gene annotations, or tissue filters. The default effect score is based on all positions in the predicted tensor:

- `delta_*` = ALT summary minus REF summary.
- `log2fc_sum` = `log2((ALT sum + pseudocount) / (REF sum + pseudocount))`.
- `max_abs_delta` and `l2_delta` require REF and ALT tensors to have the same shape.

For indels or unequal REF/ALT lengths, coordinate-aware alignment is not possible from sequences alone. The wrapper still reports aggregate differences but marks `comparable_by_position=false` and sets pointwise metrics to `NaN`.

## Output 2: all REF/ALT predictions

`all_predictions.zip` contains every prediction tensor returned by the backend for both alleles.

Example archive paths:

```text
manifest.tsv
predictions/var_1/ref/atac__1.npy
predictions/var_1/alt/atac__1.npy
predictions/var_1/ref/atac__128.npy
predictions/var_1/alt/atac__128.npy
predictions/var_1/ref/contact_maps__native.npy
predictions/var_1/alt/contact_maps__native.npy
```

Load one array:

```python
import zipfile
import numpy as np

with zipfile.ZipFile('results/all_predictions.zip') as zf:
    with zf.open('predictions/var_1/ref/atac__128.npy') as f:
        arr = np.load(f)
print(arr.shape)
```

`manifest.tsv` records input ID, allele, head, resolution, array shape, dtype, and archive path.

## Important scaling warning

AlphaGenome can produce large output tensors. Saving all raw predictions for many long sequences can create very large archives. For example, 1 bp resolution outputs scale with sequence length and number of tracks. Start with `--batch-size 1`, short sequences, and a small pilot file before running a large batch.

To reduce archive size, you can store raw predictions as float16:

```bash
python -m ag_variant_effects ... --prediction-dtype float16
```

The variant-effect table is always computed from float32 NumPy arrays after model output conversion.

## Notes and limitations

- This wrapper compares supplied REF and ALT sequences directly. It does not fetch reference genome windows, construct alternate haplotypes, or infer variant coordinates.
- Official AlphaGenome recommended variant scoring can use genomic masks, gene annotations, and modality-specific aggregation. Those require additional metadata that is not present in a bare REF/ALT sequence table.
- Track metadata is included when the backend exposes it, for example through named outputs in `alphagenome_pytorch`. Otherwise `track_name` is blank and `track_index` is used.
- The included real backend assumes channel-last AlphaGenome output tensors, which is the documented PyTorch port format.

## CLI reference

```bash
python -m ag_variant_effects --help
```

Common options:

```text
--input PATH                 CSV/TSV input file
--input-sep auto|csv|tsv     Input separator handling
--ref-col NAME               Column containing REF sequences
--alt-col NAME               Column containing ALT sequences
--id-col NAME                Optional sequence/variant ID column
--backend mock|alphagenome-pytorch
--weights PATH               Local model artifact path for real backend
--organism human|mouse       Organism index passed to the model
--device cpu|cuda            Torch device
--batch-size INT             Groups equal-length sequence pairs together
--effects-out PATH           Variant effects output table
--predictions-out PATH       Raw prediction zip archive
--prediction-dtype float32|float16
--ambiguous zeros|uniform|raise
--pseudocount FLOAT          Pseudocount for log2 fold-change summaries
--include-padding            Ask named backend to include padded tracks when supported
```
