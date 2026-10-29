"""Pre-download detector weights into models/detector (offline-friendly startup)."""

from ml import config as cfg


def main() -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=cfg.MODEL_ID, local_dir=str(cfg.LOCAL_MODEL_DIR))
    print(f"Model ready at: {path}")


if __name__ == "__main__":
    main()
