#!/usr/bin/env python3
"""Split the MLX safetensors shards into <2 GiB parts for GitHub Releases.

GitHub caps individual files at 2 GiB for both Git LFS and Release assets, and
these shards are ~5 GB each, so each is cut into three byte-range parts. The
parts concatenate back to a byte-identical copy of the Hugging Face original.

Single streaming pass: reads each shard once, hashing the whole file and each
part as it writes them, so a 5 GB shard is never held in memory.

Also rewrites the generated hash block in assemble.sh so the client-side
assembler can verify every part independently and self-heal bad downloads.

Usage:
    split_weights.py --src DIR --out DIR [--assembler PATH] [--verify-only]

Exits non-zero if any source file's hash disagrees with the published
Hugging Face hash, so a corrupted local download is caught before publishing.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path

# Whole-file sha256 of each shard as published on Hugging Face.
# Source: mlx-community/Qwen3.5-9B-MLX-8bit @ main
SHARDS: dict[str, str] = {
    "model-00001-of-00002.safetensors": (
        "0dcb3cdba0f43743875c861792685da5266aebcb58f7c0e345b9cd090bb0d289"
    ),
    "model-00002-of-00002.safetensors": (
        "5abf861e7a13e7af805105270b2648634b41fda02238ae8ee1bd64628acce9b1"
    ),
}

PARTS_PER_SHARD = 3
CHUNK = 8 << 20  # 8 MiB

BEGIN = "# --- BEGIN GENERATED HASHES ---"
END = "# --- END GENERATED HASHES ---"


def split_shard(src: Path, out_dir: Path, parts: int) -> tuple[str, list[tuple[str, str, int]]]:
    """Split one shard; return (whole_file_hash, [(part_name, part_hash, size)])."""
    size = src.stat().st_size
    part_size = math.ceil(size / parts)

    whole = hashlib.sha256()
    results: list[tuple[str, str, int]] = []

    with src.open("rb") as fh:
        for idx in range(parts):
            part_name = f"{src.name}.part-{idx}"
            ph = hashlib.sha256()
            written = 0
            dest = out_dir / part_name
            with dest.open("wb") as out:
                remaining = min(part_size, size - idx * part_size)
                while remaining > 0:
                    chunk = fh.read(min(CHUNK, remaining))
                    if not chunk:
                        raise IOError(f"{src} ended early at part {idx}")
                    out.write(chunk)
                    ph.update(chunk)
                    whole.update(chunk)
                    written += len(chunk)
                    remaining -= len(chunk)
            results.append((part_name, ph.hexdigest(), written))

    return whole.hexdigest(), results


def render_block(
    shard_hashes: dict[str, str],
    part_hashes: dict[str, str],
) -> str:
    lines = [BEGIN]
    lines.append("# Regenerate with scripts/split_weights.py after re-splitting. Do not edit by hand.")

    lines.append("shard_expected_sha() {")
    lines.append('  case "$1" in')
    for name, digest in shard_hashes.items():
        lines.append(f'    {name}) echo "{digest}" ;;')
    lines.append('    *) echo "unknown shard: $1" >&2; return 1 ;;')
    lines.append("  esac")
    lines.append("}")
    lines.append("")

    lines.append("part_expected_sha() {")
    lines.append('  case "$1" in')
    for name, digest in part_hashes.items():
        lines.append(f'    {name}) echo "{digest}" ;;')
    lines.append('    *) echo "unknown part: $1" >&2; return 1 ;;')
    lines.append("  esac")
    lines.append("}")
    lines.append(END)
    return "\n".join(lines)


def patch_assembler(path: Path, block: str) -> None:
    text = path.read_text()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"markers not found in {path}")
    path.write_text(pattern.sub(lambda _: block, text, count=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="dir holding the original shards")
    ap.add_argument("--out", required=True, type=Path, help="dir to write parts into")
    ap.add_argument("--assembler", type=Path, default=None, help="assemble.sh to patch")
    ap.add_argument("--verify-only", action="store_true", help="hash sources, write nothing")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for name in SHARDS:
        if not (args.src / name).is_file():
            print(f"missing source shard: {args.src / name}", file=sys.stderr)
            return 1

    shard_hashes: dict[str, str] = {}
    part_hashes: dict[str, str] = {}

    for name, expected in SHARDS.items():
        src = args.src / name
        size = src.stat().st_size
        print(f"==> {name} ({size:,} bytes)")

        if args.verify_only:
            h = hashlib.sha256()
            with src.open("rb") as fh:
                for chunk in iter(lambda: fh.read(CHUNK), b""):
                    h.update(chunk)
            digest = h.hexdigest()
            ok = digest == expected
            print(f"    {'ok  ' if ok else 'FAIL'} {digest}")
            if not ok:
                print(f"    expected {expected}", file=sys.stderr)
                return 1
            continue

        digest, parts = split_shard(src, args.out, PARTS_PER_SHARD)
        if digest != expected:
            print(
                f"    FAIL source hash mismatch\n"
                f"    expected {expected}\n"
                f"    actual   {digest}\n"
                f"    Refusing to publish parts from a corrupt source.",
                file=sys.stderr,
            )
            return 1

        shard_hashes[name] = digest
        for part_name, part_digest, written in parts:
            if written > 2 * 1024**3:
                print(f"    FAIL {part_name} is {written} bytes, over the 2 GiB cap", file=sys.stderr)
                return 1
            part_hashes[part_name] = part_digest
            print(f"    {part_name}  {written:>13,} bytes  {part_digest[:16]}...")

    if args.verify_only:
        print("\nAll source shards match the published Hugging Face hashes.")
        return 0

    manifest = args.out / "MANIFEST.sha256"
    with manifest.open("w") as fh:
        fh.write("# Qwen3.5-9B-MLX-8bit — release part checksums (sha256)\n")
        fh.write("# Reassemble with assemble.sh; parts concatenate in part-N order.\n\n")
        for name, digest in part_hashes.items():
            fh.write(f"{digest}  {name}\n")
        fh.write("\n# Reassembled whole-file hashes (match Hugging Face)\n")
        for name, digest in shard_hashes.items():
            fh.write(f"{digest}  {name}\n")
    print(f"\n==> wrote {manifest}")

    if args.assembler:
        patch_assembler(args.assembler, render_block(shard_hashes, part_hashes))
        print(f"==> patched {args.assembler}")

    print(f"\n{len(part_hashes)} parts in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
