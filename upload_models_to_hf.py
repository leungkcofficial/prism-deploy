"""
upload_models_to_hf.py — Upload trained PRISM model artefacts to Hugging Face Hub.

Run this ONCE from the prism_deploy/ directory after populating models/ via setup_models.sh.
Requires a Hugging Face account and a Write token from huggingface.co/settings/tokens.

Usage:
    python upload_models_to_hf.py --repo <your-username>/prism-models --token hf_...
    python upload_models_to_hf.py --repo <your-username>/prism-models --private --token hf_...

    # Or set the token via environment variable:
    HF_TOKEN=hf_... python upload_models_to_hf.py --repo <your-username>/prism-models
"""
import argparse
import os

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="leungkc/prism-models",
                        help="HuggingFace repo id, e.g. myorg/prism-models")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repo (useful during peer review)")
    parser.add_argument("--token", default=None,
                        help="HF write token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Hugging Face token required.")
        print("  Pass --token hf_...  or set  HF_TOKEN=hf_...  in the environment.")
        raise SystemExit(1)

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.isdir(models_dir):
        print(f"ERROR: models/ not found at {models_dir}")
        print("Run ./setup_models.sh first to populate the models directory.")
        raise SystemExit(1)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )
    print(f"Repository: https://huggingface.co/datasets/{args.repo}")

    print(f"\nUploading {models_dir} …  (this may take 10–30 min for ~4.3 GB)\n")
    api.upload_folder(
        folder_path=models_dir,
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Upload trained PRISM model artefacts",
        ignore_patterns=["*.gitkeep"],
    )

    print("\nUpload complete.")
    print(f"  Repo URL : https://huggingface.co/datasets/{args.repo}")
    print(f"  Update Dockerfile ARG HF_MODELS_REPO={args.repo} if repo name differs.")


if __name__ == "__main__":
    main()
