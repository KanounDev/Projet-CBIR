"""Extract normalized VGG16 features from the MIRFLICKR thumbnails."""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.descriptors.vgg import build_extractor, embed, pil_to_uint8

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "mirflickr" / "raw"
OUTPUT_PATH = DATA_DIR / "features" / "vgg16_gap512_1M.npy"
THUMB_RE = re.compile(r"^im(\d+)\.jpg$", re.IGNORECASE)
EXPECTED_COUNT = 1_000_000


def find_thumbnail_paths(expected_count: int) -> list[Path]:
    paths: list[Path | None] = [None] * expected_count
    for root, _, filenames in os.walk(RAW_DIR):
        for filename in filenames:
            match = THUMB_RE.match(filename)
            if not match:
                continue
            image_id = int(match.group(1)) - 1
            if 0 <= image_id < expected_count:
                if paths[image_id] is not None:
                    raise ValueError(f"Duplicate thumbnail for image ID {image_id}")
                paths[image_id] = Path(root) / filename

    missing = [image_id for image_id, path in enumerate(paths) if path is None]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} MIRFLICKR thumbnails; first missing IDs: {missing[:10]}"
        )
    return [path for path in paths if path is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    paths = find_thumbnail_paths(EXPECTED_COUNT)
    DATA_DIR.joinpath("features").mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".tmp")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading the same ImageNet VGG16 weights used for uploaded-image searches...")
    extractor = build_extractor(device=device)
    features = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_COUNT, 512),
    )

    try:
        for start in range(0, EXPECTED_COUNT, args.batch_size):
            end = min(start + args.batch_size, EXPECTED_COUNT)
            images = []
            for path in paths[start:end]:
                with Image.open(path) as image:
                    images.append(pil_to_uint8(image))
            batch = torch.stack(images)
            features[start:end] = embed(
                extractor,
                batch,
                device=device,
                fp16=device == "cuda",
            )
            if end % 10_000 == 0 or end == EXPECTED_COUNT:
                print(f"  Embedded {end:,}/{EXPECTED_COUNT:,} thumbnails")
        features.flush()
    except Exception:
        del features
        temporary_path.unlink(missing_ok=True)
        raise
    del features
    os.replace(temporary_path, OUTPUT_PATH)

    print(f"Saved normalized VGG16 features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
