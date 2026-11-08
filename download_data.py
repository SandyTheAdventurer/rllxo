"""Download the SRSD-Feynman datasets used here (Matsubara et al., 2024) from Hugging Face into dataset/
and check them against dataset_sha256.txt.

Usage: python download_data.py
Needs huggingface_hub (pip install huggingface_hub). Only medium (the test equations) and easy (one
metadata file, for the development equations) are used.
"""
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# The dataset revisions whose files dataset_sha256.txt lists
REVISIONS = {"medium": "de7af9e2638494678ebe18d1174bbc6a02037eec", "easy": "78a2026e61eaa000fb7ed5e30f0639329e912f9d"}


def main():
    from huggingface_hub import snapshot_download
    for name in ("medium", "easy"):
        dest = ROOT / "dataset" / f"srsd_{name}"
        tmp = ROOT / "dataset" / f".hf_srsd_{name}"
        snapshot_download(repo_id=f"yoshitomo-matsubara/srsd-feynman_{name}", repo_type="dataset", local_dir=tmp,
                          revision=REVISIONS[name],
                          allow_patterns=["supp_info.json", "train/*", "test/*"])
        dest.mkdir(parents=True, exist_ok=True)
        for p in tmp.rglob("*"):
            rel = p.relative_to(tmp)
            if p.is_file() and rel.parts[0] != ".cache":
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest / rel)
        shutil.rmtree(tmp)
    bad = 0
    for line in (ROOT / "dataset_sha256.txt").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        p = ROOT / rel
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            print("MISMATCH", rel)
            bad += 1
    print("all files match dataset_sha256.txt" if not bad else f"{bad} files do not match")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
