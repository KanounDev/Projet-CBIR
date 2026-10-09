"""
Given an image id, print the 5 most similar images according to VGG
(brute force on the 1M descriptors) and save a picture of the result.

Rule used everywhere in the project: id = image number - 1   (id 0 <-> im1.jpg)

Run from the project root (cbir-project/):
    python backend/scripts/05_vgg_similar.py 12345
    python backend/scripts/05_vgg_similar.py 12345 --k 5 --out result.png
    python backend/scripts/05_vgg_similar.py --image path/to/any_picture.jpg   # external query image
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ---------- paths (relative to the project root) ----------
ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "features" / "vgg16_gap512_1M.npy"
WEIGHTS = ROOT / "data" / "models" / "vgg16_features_imagenet.pth"
RAW = ROOT / "data" / "mirflickr" / "raw"

THUMB_ROOTS = [  # (last image number covered, folder)
    (330_000, RAW / "thumbs1-33" / "thumbnails"),
    (660_000, RAW / "thumbs34-66" / "thumbnails"),
    (1_000_000, RAW / "thumbs67-100" / "thumbnails"),
]


def find_thumb(img_id: int) -> Path:
    """Path of im<id+1>.jpg. Tries the usual layout first, then a recursive search."""
    n = img_id + 1
    for last, root in THUMB_ROOTS:
        if n <= last:
            d = (n - 1) // 10_000
            for sub in (d + 1, d):                    # folders are numbered from 1 (fallback: from 0)
                guess = root / str(sub) / f"im{n}.jpg"
                if guess.exists():
                    return guess
            hits = glob.glob(str(root / "**" / f"im{n}.jpg"), recursive=True)
            if hits:
                return Path(hits[0])
    raise FileNotFoundError(f"thumbnail im{n}.jpg not found under {RAW}")


def query_from_external_image(path: str) -> np.ndarray:
    """VGG vector (512,) of any image, using the same preprocessing as the database."""
    sys.path.insert(0, str(ROOT / "backend"))
    from app.descriptors.vgg import build_extractor, embed, pil_to_uint8

    extractor = build_extractor(str(WEIGHTS), device="cpu")
    x = pil_to_uint8(Image.open(path)).unsqueeze(0)
    return embed(extractor, x, device="cpu", fp16=False)[0]


def make_montage(query_img: Image.Image, results, out_path: str, tile=192):
    """Query on the left, then the results, thumbnails enlarged for readability."""
    items = [("QUERY", query_img)] + results
    canvas = Image.new("RGB", (tile * len(items), tile + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(items):
        canvas.paste(img.convert("RGB").resize((tile, tile), Image.BICUBIC), (i * tile, 0))
        draw.text((i * tile + 4, tile + 6), label, fill="black")
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_id", nargs="?", type=int, help="image id (= image number - 1)")
    ap.add_argument("--image", help="use an external image as the query instead of an id")
    ap.add_argument("--k", type=int, default=5, help="number of similar images")
    ap.add_argument("--out", default=None, help="output picture (default: vgg_result.png)")
    args = ap.parse_args()
    if args.image_id is None and not args.image:
        ap.error("give an image id or --image")

    # memory-mapped: the 2 GB file is read from disk, not loaded all at once
    X = np.load(FEATURES, mmap_mode="r")
    assert X.shape == (1_000_000, 512), f"unexpected shape {X.shape}"

    if args.image:
        q = query_from_external_image(args.image)
        query_img, exclude = Image.open(args.image), None
        title = args.image
    else:
        if not 0 <= args.image_id < X.shape[0]:
            ap.error("image id must be between 0 and 999999")
        q = np.asarray(X[args.image_id])
        query_img, exclude = Image.open(find_thumb(args.image_id)), args.image_id
        title = f"id {args.image_id} (im{args.image_id + 1}.jpg)"

    # brute force: vectors are L2-normalized, so dot product = cosine similarity
    sims = X @ q
    if exclude is not None:
        sims[exclude] = -np.inf                       # do not return the query itself
    top = np.argpartition(-sims, args.k)[: args.k]
    top = top[np.argsort(-sims[top])]                 # best first

    print(f"Query: {title}")
    print(f"{'rank':<5}{'id':<10}{'image':<16}{'cosine sim.':<12}")
    results = []
    for rank, i in enumerate(top, 1):
        print(f"{rank:<5}{i:<10}{'im' + str(i + 1) + '.jpg':<16}{sims[i]:.4f}")
        results.append((f"#{rank} id={i}  {sims[i]:.3f}", Image.open(find_thumb(int(i)))))

    out = args.out or "vgg_result.png"
    make_montage(query_img, results, out)
    print(f"Picture saved: {out}")


if __name__ == "__main__":
    main()