"""Pre-seed the local HF cache with a registered model's weights, CPU-only.

Useful before a run with HF_HUB_OFFLINE=1 (no network access at run time) --
pre-downloading avoids a LocalEntryNotFoundError later for a model that was
never cached.

Uses huggingface_hub.snapshot_download (file transfer only, no
AutoModelForCausalLM.from_pretrained) so a large model downloads without
materializing the state dict in RAM -- this job needs disk + network
bandwidth, not GPU or much CPU memory. --model resolves --repo-id + gated
license status from re_polar.models.MODEL_REGISTRY so it can never drift from
what the rest of the pipeline will actually try to load.

Usage:
  python -m re_polar.datasets.download_weights --model qwen3_32b
  python -m re_polar.datasets.download_weights --repo-id Qwen/Qwen3-32B  # ungated escape hatch
"""

import argparse

from re_polar.models import MODEL_REGISTRY

# original (non-safetensors) weight formats some repos ship alongside
# safetensors -- skip them, they're never loaded (re_polar/core/model_loader.py always
# takes the safetensors path) and roughly double the download for nothing.
IGNORE_PATTERNS = ["*.bin", "*.msgpack", "*.h5", "*.ot", "*.pth", "original/*"]


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY),
        help="resolves repo id via MODEL_REGISTRY[model]['model_id']",
    )
    group.add_argument("--repo-id", help="raw HF repo id, bypassing MODEL_REGISTRY")
    args = parser.parse_args()

    repo_id = MODEL_REGISTRY[args.model]["model_id"] if args.model else args.repo_id

    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} into HF_HOME cache (ignoring {IGNORE_PATTERNS}) ...")
    path = snapshot_download(repo_id=repo_id, ignore_patterns=IGNORE_PATTERNS)
    print(f"Done -> {path}")


if __name__ == "__main__":
    main()
