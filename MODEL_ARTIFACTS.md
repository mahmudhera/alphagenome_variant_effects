# Model artifact instructions

This wrapper does **not** ship AlphaGenome weights or the AlphaGenome architecture.

It provides batch input/output, generic REF/ALT effect summaries, and a backend adapter for a local PyTorch AlphaGenome implementation.

## Why there are multiple artifact paths

Google DeepMind's official open-source AlphaGenome research repository is implemented in JAX. It provides factory methods such as `dna_model.create_from_kaggle('all_folds')` and `dna_model.create_from_huggingface('all_folds')`, but running that code requires a JAX-capable environment.

Your stated target environment is PyTorch/TensorFlow/NumPy/pandas/SciPy with no Hugging Face Python stack. For that reason, this wrapper targets a local PyTorch checkpoint through `alphagenome_pytorch.AlphaGenome.from_pretrained(local_path)`.

## Path A: use a PyTorch-converted checkpoint

Place a local artifact here:

```text
artifacts/model_all_folds.safetensors
```

or any other artifact name accepted by your installed/vendored `alphagenome_pytorch` version:

```text
artifacts/alphagenome.pt
artifacts/fold_1_weights.pth
```

Run:

```bash
PYTHONPATH=src python -m ag_variant_effects \
  --backend alphagenome-pytorch \
  --weights artifacts/model_all_folds.safetensors \
  --input examples/input.tsv \
  --ref-col ref_seq \
  --alt-col alt_seq \
  --id-col variant_id \
  --effects-out results/variant_effects.tsv \
  --predictions-out results/all_predictions.zip
```

### Getting the PyTorch implementation without Hugging Face Python libraries

Install from PyPI/GitHub if your environment allows it:

```bash
python -m pip install alphagenome-pytorch
# or
python -m pip install git+https://github.com/genomicsxai/alphagenome-pytorch
```

Or vendor it without installing:

```bash
mkdir -p third_party
git clone https://github.com/genomicsxai/alphagenome-pytorch third_party/alphagenome-pytorch
PYTHONPATH=src:third_party/alphagenome-pytorch/src python -m ag_variant_effects --help
```

The wrapper never imports `huggingface_hub` or `transformers`.

### Downloading a checkpoint without Hugging Face Python libraries

Using `curl`:

```bash
export HF_TOKEN='hf_...'
bash scripts/download_hf_without_python_hf.sh model_all_folds.safetensors artifacts/model_all_folds.safetensors
```

Using a browser:

1. Open the model repository in a browser.
2. Accept the applicable model terms/license if prompted.
3. Download the desired checkpoint file.
4. Copy it into `artifacts/`.

If `.safetensors` loading is unsupported in your runtime, convert the checkpoint to `.pt`/`.pth` in a separate environment where `safetensors` is installed, then copy the converted file into `artifacts/`.

## Path B: official Google DeepMind JAX weights from Kaggle

Use this path if you have a JAX-capable machine.

1. Open the official AlphaGenome Research repository instructions.
2. Accept the model terms.
3. Download the pretrained model weights from Kaggle.
4. Run the official JAX code directly, or convert the JAX checkpoint to a PyTorch checkpoint using the conversion utilities from the PyTorch port.
5. Copy the converted PyTorch checkpoint into `artifacts/` for this wrapper.

## Path C: hosted AlphaGenome API

If you do not need local artifacts, the hosted API is the simplest route for moderate-scale analysis. This wrapper is local-model oriented, so the hosted API is not implemented as a backend here.

## Expected local files

For this wrapper plus the PyTorch backend:

```text
artifacts/model_all_folds.safetensors  # or .pt/.pth accepted by AlphaGenome.from_pretrained
```

Optional:

```text
third_party/alphagenome-pytorch/src/alphagenome_pytorch/
```

## Artifact/license reminder

AlphaGenome model parameters, model outputs, and derivatives can be subject to Google DeepMind model terms. Keep the downloaded artifacts out of git. This repo's `.gitignore` excludes `artifacts/*`, `*.pt`, `*.pth`, and `*.safetensors`.
