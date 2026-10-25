"""Create missing CBIR features, indexes, metadata, and MongoDB records."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "mirflickr" / "raw"
EXPECTED_COUNT = 1_000_000
DESCRIPTOR_FEATURES = {
    "edgehistogram": ("edgehistogram.npy", "edgehistogram_ids.npy", 150),
    "homogeneoustexture": ("homogeneoustexture.npy", "homogeneoustexture_ids.npy", 43),
}
REQUIRED_RAW_DIRS = (
    "thumbs1-33",
    "thumbs34-66",
    "thumbs67-100",
    "tags",
    "license",
    "features_edgehistogram",
    "features_homogeneoustexture",
)
REQUIRED_ASSETS = (
    DATA_DIR / "features" / "vgg16_gap512_1M.npy",
    DATA_DIR / "indexes" / "vgg16_euclidean.faiss",
    DATA_DIR / "indexes" / "edgehistogram_euclidean.faiss",
    DATA_DIR / "indexes" / "homogeneoustexture_euclidean.faiss",
    DATA_DIR / "metadata" / "metadata.jsonl",
)


def run_script(name: str, *args: str) -> None:
    script = ROOT / "backend" / "scripts" / name
    subprocess.run([sys.executable, str(script), *args], cwd=ROOT, check=True)


def validate_raw_data() -> None:
    missing = [str(RAW_DIR / name) for name in REQUIRED_RAW_DIRS if not (RAW_DIR / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            "Required MIRFLICKR raw-data directories are missing:\n  "
            + "\n  ".join(missing)
            + "\nDownload and extract all MIRFLICKR archives into data/mirflickr/raw."
        )


def validate_descriptor(name: str) -> bool:
    feature_name, ids_name, dimension = DESCRIPTOR_FEATURES[name]
    feature_path = DATA_DIR / "features" / feature_name
    ids_path = DATA_DIR / "features" / ids_name
    if not feature_path.is_file() or not ids_path.is_file():
        return False

    features = np.load(feature_path, mmap_mode="r")
    ids = np.load(ids_path, mmap_mode="r")
    if features.shape != (EXPECTED_COUNT, dimension):
        raise ValueError(f"Unexpected shape for {feature_path}: {features.shape}")
    if ids.shape != (EXPECTED_COUNT,) or not np.array_equal(
        ids, np.arange(EXPECTED_COUNT, dtype=ids.dtype)
    ):
        raise ValueError(f"Image IDs in {ids_path} must be contiguous from 0 to 999999")
    return True


def validate_required_assets() -> list[Path]:
    missing = [
        path
        for path in REQUIRED_ASSETS
        if not path.is_file() or path.stat().st_size == 0
    ]
    for name in DESCRIPTOR_FEATURES:
        feature_name, ids_name, _ = DESCRIPTOR_FEATURES[name]
        for filename in (feature_name, ids_name):
            path = DATA_DIR / "features" / filename
            if not path.is_file():
                missing.append(path)
    return missing


def prepare() -> None:
    validate_raw_data()
    (DATA_DIR / "features").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "indexes").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "metadata").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "models").mkdir(parents=True, exist_ok=True)

    vgg_features = DATA_DIR / "features" / "vgg16_gap512_1M.npy"
    if not vgg_features.is_file():
        print("VGG16 features are missing; extracting them from all thumbnails.")
        run_script("build_vgg16_features.py")
    else:
        shape = np.load(vgg_features, mmap_mode="r").shape
        if shape != (EXPECTED_COUNT, 512):
            raise ValueError(f"Unexpected shape for {vgg_features}: {shape}")

    for name in DESCRIPTOR_FEATURES:
        if not validate_descriptor(name):
            print(f"{name} features are missing; parsing the raw MPEG-7 descriptors.")
            run_script("parse_descriptors.py", name)
            if not validate_descriptor(name):
                raise RuntimeError(f"Generated {name} features failed validation")

    metadata = DATA_DIR / "metadata" / "metadata.jsonl"
    if not metadata.is_file() or metadata.stat().st_size == 0:
        print("Metadata is missing; building it from tags, licenses, and thumbnails.")
        run_script("build_metadata.py")

    vgg_index = DATA_DIR / "indexes" / "vgg16_euclidean.faiss"
    if not vgg_index.is_file() or vgg_index.stat().st_size == 0:
        run_script("build_vgg16_index.py")

    for name in DESCRIPTOR_FEATURES:
        index = DATA_DIR / "indexes" / f"{name}_euclidean.faiss"
        if not index.is_file() or index.stat().st_size == 0:
            run_script("build_indexes.py", "--faiss", "--descriptors", name)

    run_script("load_mongo.py", "--if-empty", "--expected-count", str(EXPECTED_COUNT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="check raw data and generated assets without creating or loading anything",
    )
    args = parser.parse_args()

    validate_raw_data()
    if args.check_only:
        missing = validate_required_assets()
        vgg_features = DATA_DIR / "features" / "vgg16_gap512_1M.npy"
        if vgg_features.is_file():
            shape = np.load(vgg_features, mmap_mode="r").shape
            if shape != (EXPECTED_COUNT, 512):
                raise ValueError(f"Unexpected shape for {vgg_features}: {shape}")
        metadata = DATA_DIR / "metadata" / "metadata.jsonl"
        if not metadata.is_file() or metadata.stat().st_size == 0:
            missing.append(metadata)
        for name in DESCRIPTOR_FEATURES:
            if not validate_descriptor(name):
                missing.append(DATA_DIR / "features" / DESCRIPTOR_FEATURES[name][0])
        if missing:
            print("Missing generated assets:")
            for path in missing:
                print(f"  {path}")
            raise SystemExit(1)
        print("Raw data and required generated assets are present.")
        return

    prepare()
    print("CBIR data preparation complete.")


if __name__ == "__main__":
    main()
