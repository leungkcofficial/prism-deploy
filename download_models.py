"""
download_models.py — Fetch trained PRISM model artefacts from Hugging Face Hub.

Use this script to populate models/ locally without Docker, e.g. for running
the replication notebook or the FastAPI app directly with uvicorn.

Usage:
    python download_models.py                          # default HF repo
    HF_MODELS_REPO=myorg/my-repo python download_models.py  # custom repo
"""
import os
import sys

HF_REPO = os.environ.get("HF_MODELS_REPO", "leungkc/prism-models")
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "models")


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading models from {HF_REPO} → {LOCAL_DIR}")
    print("(This is a one-time download of ~4.3 GB. Docker caches this layer.)\n")

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", ".gitattributes"],
    )

    print("\nDone. Model files:")
    for root, dirs, files in os.walk(LOCAL_DIR):
        dirs.sort()
        for f in sorted(files):
            path = os.path.join(root, f)
            size_mb = os.path.getsize(path) / 1e6
            rel = os.path.relpath(path, LOCAL_DIR)
            print(f"  {rel:55s}  {size_mb:7.1f} MB")


if __name__ == "__main__":
    main()
