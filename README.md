---
base_model: Qwen/Qwen3.5-9B
library_name: mlx
tags:
  - mlx
  - qwen3.5
  - vision-language-model
  - quantized
  - 8bit
license: apache-2.0
---

# Qwen3.5-9B-MLX-8bit

This is a quantized MLX version of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) for Apple Silicon.

> **This is a mirror.** The canonical, authoritative copy lives at
> [`mlx-community/Qwen3.5-9B-MLX-8bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-8bit)
> on Hugging Face. This repository exists to serve the same weights from
> GitHub Releases. If you just want to use the model, load it from Hugging
> Face directly — that is simpler and always up to date:
>
> ```python
> from mlx_vlm import load
> model, processor = load("mlx-community/Qwen3.5-9B-MLX-8bit")
> ```

## Getting the weights

The config and tokenizer files are in this repository. The two weight shards
are ~5 GB each, which is over GitHub's 2 GiB per-file cap for both Git LFS and
Release assets, so each shard is published as **three byte-range parts** in the
[`weights-v1` release](../../releases/tag/weights-v1). Concatenating the parts
reproduces the original files byte for byte.

```bash
git clone https://github.com/leo-kreisman/Qwen3.5-9B-MLX-8bit.git
cd Qwen3.5-9B-MLX-8bit
./assemble.sh
```

The script downloads the six parts, verifies each one against its published
sha256, concatenates them, and checks the reassembled shards against the
hashes from Hugging Face. When it finishes, the checkout is a loadable model
directory. It is resumable — re-running it skips parts already downloaded and
verified, and re-fetches any part that fails its checksum.

| Release asset | Bytes | sha256 (first 16) |
| --- | --- | --- |
| `model-00001-of-00002.safetensors.part-0` | 1,779,840,842 | `acef14c2b9d677ea` |
| `model-00001-of-00002.safetensors.part-1` | 1,779,840,842 | `bd625308afd9a066` |
| `model-00001-of-00002.safetensors.part-2` | 1,779,840,841 | `4c669cb2665f5def` |
| `model-00002-of-00002.safetensors.part-0` | 1,695,689,966 | `b54e5cbbbb759bbf` |
| `model-00002-of-00002.safetensors.part-1` | 1,695,689,966 | `1fdb0439249dcb12` |
| `model-00002-of-00002.safetensors.part-2` | 1,695,689,966 | `1ba781142f9db3b4` |

Full hashes are in [`MANIFEST.sha256`](../../releases/tag/weights-v1) and are
also pinned inside `assemble.sh`, so the script verifies every part and can
re-fetch any that fails.

Assembled shard hashes (identical to Hugging Face):

```
0dcb3cdba0f43743875c861792685da5266aebcb58f7c0e345b9cd090bb0d289  model-00001-of-00002.safetensors
5abf861e7a13e7af805105270b2648634b41fda02238ae8ee1bd64628acce9b1  model-00002-of-00002.safetensors
```

To do it by hand instead of using the script:

```bash
BASE=https://github.com/leo-kreisman/Qwen3.5-9B-MLX-8bit/releases/download/weights-v1
for n in 00001 00002; do
  f=model-$n-of-00002.safetensors
  curl -fLO "$BASE/$f.part-0"
  curl -fLO "$BASE/$f.part-1"
  curl -fLO "$BASE/$f.part-2"
  cat $f.part-0 $f.part-1 $f.part-2 > $f
  shasum -a 256 $f    # compare against the hashes above
done
```

## Model Details

- **Original Model:** [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Quantization:** 8-bit (~8.864 bits per weight)
- **Group Size:** 64
- **Format:** MLX SafeTensors
- **Framework:** [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)

## Conversion Details

This model was converted using `mlx-vlm` from the [`pc/fix-qwen35-predicate`](https://github.com/Blaizzy/mlx-vlm/tree/pc/fix-qwen35-predicate) branch, which includes fixes for Qwen3.5 quantization predicates (proper handling of MoE gate layers, `shared_expert_gate`, and `A_log` casting).

**Conversion command:**
```bash
python3 -m mlx_vlm convert \
  --hf-path "Qwen/Qwen3.5-9B" \
  --mlx-path "./mlx_models/Qwen3.5-9B-MLX-8bit" \
  -q --q-bits 8 --q-group-size 64
```

## Important Note

A better, more optimized conversion may be available from **@Prince** ([@Blaizzy](https://huggingface.co/Blaizzy)) in the MLX VLM community. Check the [mlx-community](https://huggingface.co/mlx-community) organization for updated versions as official Qwen3.5 support is merged into the main `mlx-vlm` branch.

## Usage

```python
from mlx_vlm import load, generate

model, processor = load("mlx-community/Qwen3.5-9B-MLX-8bit")

output = generate(
    model,
    processor,
    prompt="Describe this image in detail",
    image="path/to/image.jpg",
    max_tokens=200
)
print(output)
```

To load the local copy instead of the Hugging Face one, point `load` at this
directory (after running `./assemble.sh`):

```python
model, processor = load("/path/to/Qwen3.5-9B-MLX-8bit")
```

Or from the command line:
```bash
mlx_vlm generate \
  --model mlx-community/Qwen3.5-9B-MLX-8bit \
  --prompt "Describe this image" \
  --image path/to/image.jpg \
  --max-tokens 200
```

## Performance

- **Disk Size:** ~9.8 GB
- Runs efficiently on Apple Silicon Macs (M1/M2/M3/M4)

## License

This model inherits the [Apache 2.0 license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE) from the original Qwen3.5-9B model. The conversion was produced by
[mlx-community](https://huggingface.co/mlx-community); this repository only
redistributes it.
