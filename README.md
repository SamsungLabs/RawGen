# RawGen: Learning Camera Raw Image Generation

[Dongyoung Kim](https://www.dykim.me/)<sup>1</sup>,
[Junyong Lee](https://junyonglee.me/)<sup>1\*</sup>,
[Abhijith Punnappurath](https://abhijithpunnappurath.github.io/)<sup>1\*</sup>,
[Mahmoud Afifi](https://www.mafifi.info/)<sup>1\*</sup>,
[Sangmin Han](https://sites.google.com/view/sangmin-han/home)<sup>2</sup>,
[Alex Levinshtein](https://scholar.google.com/citations?user=7EFMKWUAAAAJ&hl=en)<sup>1</sup>,
[Michael S. Brown](https://www.eecs.yorku.ca/~mbrown/)<sup>1</sup>

<sup>1</sup>AI Center – Toronto, Samsung Electronics &nbsp;&nbsp;
<sup>2</sup>Yonsei University &nbsp;&nbsp;
<sup>\*</sup>Equal contribution

[[Paper]](https://arxiv.org/abs/2604.00093)
[[Project page]](https://dy112.github.io/rawgen-page/)

Official implementation of the ECCV 2026 paper.

## Overview

RawGen maps a photo-finished **sRGB image** or a **text prompt** back to a
standardized **CIE XYZ** representation, then renders that XYZ as camera RAW.
The XYZ intermediate is camera-agnostic, so one prediction can be re-rendered
for any camera whose colour calibration is known.

```
sRGB image ─┐
            ├─► VAE encode ─► Kontext LoRA denoise ─► fine-tuned VAE decoder ─► XYZ ─┐
text prompt ┘        (variant → anchor, in latent space)                             │
                                                                                     ▼
                                              9 cross-camera RAW PNGs + Samsung S20 / S25U DNGs
```

Two fine-tuned components sit on top of frozen FLUX.1-Kontext-dev:

- a **LoRA adapter** on the Kontext transformer, learning the variant→anchor
  mapping in latent space (rank 64, on `to_q / to_k / to_v / to_out.0`);
- a **fully fine-tuned VAE decoder** that decodes anchor latents to CIE XYZ
  instead of sRGB.

## Requirements

A CUDA GPU with ≥40 GB VRAM. bfloat16 throughout.

```bash
pip install -r requirements.txt

# system dependency: ExifTool (PyExifTool shells out to it to read DNG metadata)
#   Ubuntu:  sudo apt-get install -y libimage-exiftool-perl
#   macOS:   brew install exiftool
```

`diffusers` is pinned; do not upgrade past the pin. The bottom section of
`requirements.txt` lists training-only extras that inference does not need.

The base models `black-forest-labs/FLUX.1-Kontext-dev` and (for text-to-RAW)
`black-forest-labs/FLUX.1-dev` are **gated**. Accept each model's license on its
model page, then authenticate (`hf auth login`, or export `HF_TOKEN`) before the
first run; `diffusers` fetches them automatically from there.

## Pretrained models

Model weights are not distributed here. Train them with Steps 3–4 below, or use the
checkpoints released at https://github.com/DY112/RawGen.

## Usage

Run both CLIs from the repository root; their default paths
(`checkpoints/`, `samples/`) are relative to it.

### Image-to-RAW

Point `--input` at an sRGB `.jpg` / `.png` / `.tif` file, or at a directory of
them. Inputs are resized to 1024², the training resolution.

```bash
python image_to_raw.py \
    --input samples/inputs/ \
    --output-dir out/image_to_raw/ \
    --num-illum 2 \
    --num-steps 30 \
    --seed 42
```

### Text-to-RAW

```bash
python text_to_raw.py \
    --prompts-file samples/prompts.txt \
    --output-dir out/text_to_raw/
```

Text-to-RAW runs two diffusion stages, so its step and guidance flags are split
per stage; there is no `--num-steps` here.

#### Options

Shared by both CLIs:

| Flag | Default | Effect |
|---|---|---|
| `--num-illum` | 2 | How many illuminations to sample per image; each one produces its own set of camera outputs. Illuminants are drawn at random from `samples/illuminations.json`. The illumination used is recorded in `metadata.json` |
| `--kontext-guidance` | 3.5 | Guidance for the Kontext denoise stage; 3.5 is the trained value |
| `--seed` | 42 | Seeds both the illumination sampling and the initial denoise noise, so a rerun reproduces the outputs |
| `--transformer-ckpt` / `--vae-decoder-ckpt` | `checkpoints/…` | Load the two checkpoints from elsewhere |
| `--illum-json` / `--dng-dir` | `samples/…` | Illumination set and camera-profile DNGs |

Image-to-RAW only: `--num-steps` (30) denoise steps, `--save-jpg` additionally
writes 8-bit JPEG previews next to the RAW PNGs.

Text-to-RAW only: `--t2i-steps` (30) for the FLUX.1-dev text-to-image stage and
`--kontext-steps` (30) for the Kontext stage; `--guidance-scale` (3.5) for the
text-to-image stage; `--batch-size` (2) prompts per forward pass.

### Output layout

```
out/<task>/
├── xyz/<name>_xyz.png                    16-bit OETF-encoded XYZ
├── png/<name>_<camera>_illum<i>.png      cross-camera RAW, 9 NUS cameras
├── dng/<name>_S20_illum<i>.dng           Samsung S20 RAW (Bayer)
├── dng/<name>_S25U_illum<i>.dng          Samsung S25 Ultra RAW (ProRAW)
└── metadata.json                         checkpoints, seed, per-camera gt_illum
```

The `xyz/` image is **sRGB-OETF-encoded**, not linear. Apply the inverse OETF
before using it in any linear-domain computation.

`<camera>` is one of the nine NUS bodies (`NikonD40`, `Canon1DsMkIII`, …), each
matched to its profile DNG in `samples/dng_profiles/`. `<i>` indexes the sampled
illuminations (`0` to `--num-illum - 1`). The `png/` files are viewing previews;
the two `dng/` files are real RAW containers. `metadata.json` records the
checkpoints, seed, and the illumination vector used for each camera.

Text-to-RAW indexes outputs by prompt rather than by input filename, so `<name>`
above is `<NNNN>_<slug>`; it also writes the prompt to `text_prompt/<NNNN>_<slug>.txt`.

S25U output is Samsung ProRAW (JPEG-XL compressed, linear 3-channel). Read it
with `tifffile` + `imagecodecs`, or with Adobe software.

## Training

Training takes four steps, preceded by building the manifest they all read. All
multi-GPU scripts use `torchrun`. Run them from `training/`.

The commands below are minimal runnable examples; the released settings are
listed under [Released checkpoint settings](#released-checkpoint-settings).

### Step 0 — Build the manifest

`split_trainval.py` writes `trainval.json` from what is on disk: it pairs
`<root>/<dataset_type>/sRGB/*_srgb.png` with `<root>/<dataset_type>/XYZ/*_xyz.png`
by basename, shuffles with `--seed`, and splits by `--train-ratio` / `--val-ratio`.

```bash
python split_trainval.py \
    --root /path/to/data --output ../trainval.json \
    --train-ratio 0.8 --val-ratio 0.1 --seed 42
```

Passing `--dataset-types a5k` rewrites only those types and leaves the rest of an
existing manifest untouched. The released split used `--seed 42` and 8:1:1.

### Dataset manifest

Both data preparation and training read a `trainval.json` manifest:

```json
{
  "train": [ {"dataset_type": "a5k", "basename": "a0001-..."}, ... ],
  "val":   [ ... ],
  "test":  [ ... ],
  "meta":  { "root": "/abs/path/to/data",
             "srgb_suffix": "_srgb.png", "xyz_suffix": "_xyz.png" }
}
```

`--root-key` selects which `meta.*` root to resolve against. `--split all`
concatenates train+val+test; `--datasets a5k raise` (or `all`) selects subsets
by `dataset_type`.

#### Getting the source data

Neither dataset is redistributed here. Both are free for research use but
require their own registration/download:

| `dataset_type` | Dataset | Ships as | Prepare by |
|---|---|---|---|
| `a5k` | [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) | `.dng` | use as-is |
| `raise` | [RAISE](http://loki.disi.unitn.it/RAISE/) | `.nef` (Nikon raw) | **convert to DNG** (e.g. Adobe DNG Converter) |

`dataset_type` is the directory name, and `basename` is the DNG
filename without its extension, so a manifest row
`{"dataset_type": "a5k", "basename": "a0104-dvf_003"}` resolves to
`<meta.root>/a5k/DNG/a0104-dvf_003.dng`. The released manifest keeps each
dataset's native naming (FiveK `a0104-dvf_003`, RAISE `rc5ae823dt`); any naming
works as long as the manifest and the filenames agree.

#### Directory layout

```
<meta.root>/
├── a5k/
│   ├── DNG/<basename>.dng          # you provide — Step 1 reads these
│   ├── sRGB/<basename>_srgb.png    # you provide — Step 0 enumerates these
│   ├── XYZ/<basename>_xyz.png      # you provide — Step 0 enumerates these
│   ├── anchor-variants-imgs/       # Step 1 writes here
│   └── latents/                    # Step 2 writes here
└── raise/
    └── ... same five
```

A manifest entry may instead give explicit absolute `"srgb"` / `"xyz"` paths, but
they must keep the `<root>/<dataset_type>/<subdir>/` shape.

#### Keeping the four steps consistent

Steps 2-4 locate the earlier outputs by directory **name**, so the same name
has to be repeated:

| What | Step 1 | Step 2 | Step 3 | Step 4 |
|---|---|---|---|---|
| variant/anchor images | `--output-dir` | `--variant-dir-name` | — | `--gt-xyz-dir-name` |
| latents | — | `--output-dir` | `--latent-dir-name` | `--latent-dir-name` |
| variants per anchor | `--num-variations` | `--num-variations` | `--num-variations` | — |

The example commands below use `anchor-variants-imgs` and `latents` throughout.

### Step 1 — ISP variants and anchors from DNGs

Synthesizes N sRGB variants plus the XYZ / sRGB anchors from each source DNG,
using the vendored mini ISP simulator. The XYZ anchor runs the ISP through the
`xyz` stage and then the sRGB OETF (`--xyz-anchor-gamma`, on by default); pass
`--no-xyz-anchor-gamma` for linear anchors. The paper's reconstruction metrics
were computed in this encoded space.

```bash
python generate_variants_and_anchors.py \
    --trainval-json /path/to/trainval.json --split all --datasets a5k \
    --output-dir anchor-variants-imgs --num-variations 5 --cpu-workers 24 \
    --generate-xyz-anchor
```

### Step 2 — VAE-encode to latents

Pre-encodes the Step 1 images so training iterates without rerunning the VAE.
Re-runnable with a different VAE without redoing Step 1.

```bash
torchrun --nproc_per_node=4 generate_latents_for_variants_and_anchors.py \
    --model-id black-forest-labs/FLUX.1-dev \
    --trainval-json /path/to/trainval.json --split all --datasets a5k \
    --variant-dir-name anchor-variants-imgs \
    --output-dir latents --num-variations 5 --batch-size 8 \
    --generate-xyz-anchor
```

Produces `{basename}_anchor_srgb.pt`, `{basename}_anchor_xyz.pt` and
`{basename}_var_{ii:02d}.pt` per basename. Encoding uses `latent_dist.mode()`,
so the cache is deterministic.

### Step 3 — Kontext diffusion LoRA

```bash
torchrun --nproc_per_node=4 train_kontext_lora.py \
    --model-id black-forest-labs/FLUX.1-Kontext-dev \
    --trainval-json /path/to/trainval.json --datasets a5k \
    --latent-dir-name latents --root-key root \
    --batch-size 4 --lr 1e-4 --max-epochs 40 --save-every 5
```

The saved `.pt` holds the LoRA adapter (no FLUX base weights) plus optimizer
state for resuming; the CLIs read only the adapter.

### Step 4 — VAE decoder

```bash
torchrun --nproc_per_node=4 finetune_vae_decoder.py \
    --model-id black-forest-labs/FLUX.1-dev \
    --trainval-json /path/to/trainval.json --datasets a5k \
    --latent-dir-name latents --gt-xyz-dir-name anchor-variants-imgs \
    --amp --batch-size 8 --lr 1e-4 --max-epochs 30 --save-every 10
```

`--amp` runs the decode under autocast and is what the released run used. The
decoder trains on the FLUX.1-dev VAE and loads onto the Kontext VAE at inference.

### Using your own checkpoints

Both trainers write to a timestamped run directory under `--results-root`
(default `./results/`): Step 3 to `<run>/kontext_lora_ckpt/kontext_lora_epoch{N}.pt`
(plus `_final`), Step 4 to `<run>/vae_decoder_xyz_ckpt_finetune/vae_decoder_xyz_epoch{N}.pt`
(plus `_best` and `_final`). Pass either file to `--transformer-ckpt` /
`--vae-decoder-ckpt`, or copy them to `checkpoints/kontext_lora.pt` and
`checkpoints/vae_decoder_xyz.pt` to use the defaults.

### Released checkpoint settings

Both released checkpoints were trained on 8 GPUs over both dataset subsets
(`--datasets all`), with 5 ISP variants per anchor.

`kontext_lora.pt` — 40 epochs, LoRA rank 64 / alpha 64 on
`to_q, to_k, to_v, to_out.0`, bfloat16:

```bash
torchrun --nproc_per_node=8 train_kontext_lora.py \
    --model-id black-forest-labs/FLUX.1-Kontext-dev \
    --trainval-json /path/to/trainval.json --datasets all \
    --latent-dir-name latents --num-variations 5 \
    --amp --param-dtype bfloat16 --gradient-checkpointing \
    --batch-size 1 --grad-accum-steps 2 --lr 1e-4 --warmup-steps 200 \
    --max-epochs 40 --save-every 5
```

`vae_decoder_xyz.pt` — 30 epochs, full fine-tune, L1 loss, tighter gradient
clipping:

```bash
torchrun --nproc_per_node=8 finetune_vae_decoder.py \
    --model-id black-forest-labs/FLUX.1-dev \
    --trainval-json /path/to/trainval.json --datasets all \
    --latent-dir-name latents --gt-xyz-dir-name anchor-variants-imgs \
    --amp --dtype bfloat16 --param-dtype bfloat16 \
    --batch-size 2 --lr 1e-4 \
    --clip-grad-norm 0.5 --max-epochs 30
```

## Acknowledgements

The low-level DNG and camera-pipeline code is vendored from
[graphics2raw](https://github.com/SamsungLabs/graphics2raw) and
[simple-camera-pipeline](https://github.com/AbdoKamel/simple-camera-pipeline);
each vendored file records its origin and license in its header. The nine
reference camera profiles come from the NUS
Illumination Dataset (Cheng et al.). See `LICENSE.md` for provenance details.

## Citation

```bibtex
@inproceedings{kim2026rawgen,
  title     = {RawGen: Learning Camera Raw Image Generation},
  author    = {Dongyoung Kim and Junyong Lee and Abhijith Punnappurath and
               Mahmoud Afifi and Sangmin Han and Alex Levinshtein and
               Michael S. Brown},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

Code written for RawGen is **CC BY-NC-SA 4.0**. Vendored third-party code keeps
its own upstream license, and the model weights are governed by the FLUX.1 [dev]
Non-Commercial License. See [`LICENSE.md`](LICENSE.md) for the per-path
breakdown and [`NOTICE`](NOTICE) for the weight attribution.
