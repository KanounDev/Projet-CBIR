"""Step 4a - merge tags + license + thumbnail path into one file: data/metadata/metadata.jsonl

Usage (from C:\\projet-CBIR, environment active):
    python backend/scripts/04_build_metadata.py --limit 1000    # quick test
    python backend/scripts/04_build_metadata.py                 # full run

One JSON object per line, one image per object. id = file number (0..999999),
image number = id + 1 (im<id+1>.jpg). Missing fields are null.
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "mirflickr" / "raw"
OUT = ROOT / "data" / "metadata"
TXT_RE = re.compile(r"^(\d+)\.txt$")
JPG_RE = re.compile(r"^im(\d+)\.jpg$", re.IGNORECASE)

LICENSE_KEYS = {
    "Date uploaded": "date_uploaded",
    "License": "license",
    "License url": "license_url",
    "Picture title": "title",
    "Owner name": "owner_name",
    "Owner username": "owner_username",
    "Owner id": "owner_id",
    "Photo id": "photo_id",
    "Photo url": "photo_url",
    "Web url": "web_url",
}


def scan(folder, regex, shift=0):
    found = {}
    for root, _, files in os.walk(folder):
        for f in files:
            m = regex.match(f)
            if m:
                found[int(m.group(1)) + shift] = os.path.join(root, f)
    return found


def read_text(path):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def parse_license(path):
    out = {v: None for v in LICENSE_KEYS.values()}
    for line in read_text(path).splitlines():
        key, sep, value = line.partition(":")
        key = key.strip()
        if sep and key in LICENSE_KEYS:
            out[LICENSE_KEYS[key]] = value.strip() or None
    return out


def parse_tags(path):
    seen, tags = set(), []
    for line in read_text(path).splitlines():
        t = line.strip().lower()
        if t and t not in seen:
            seen.add(t)
            tags.append(t)
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    print("Scanning folders ...")
    lic = scan(RAW / "license", TXT_RE)
    tag = scan(RAW / "tags", TXT_RE)
    thumbs = {}
    for d in sorted(p for p in RAW.iterdir() if p.is_dir() and p.name.startswith("thumbs")):
        thumbs.update(scan(d, JPG_RE, shift=-1))  # im1.jpg -> id 0
    ids = sorted(set(lic) | set(tag) | set(thumbs))
    if args.limit:
        ids = ids[: args.limit]
    elif ids != list(range(1_000_000)):
        raise ValueError(
            "Expected metadata IDs 0..999999 from the complete tags, license, "
            "and thumbnail data. Check that all MIRFLICKR raw archives are extracted."
        )
    print(f"license={len(lic)} tags={len(tag)} thumbs={len(thumbs)} -> {len(ids)} ids ({time.time() - t0:.0f}s)")

    def build(i):
        rec = {"id": i, "thumb": None, "keywords": [], **{v: None for v in LICENSE_KEYS.values()}}
        if i in thumbs:
            rec["thumb"] = os.path.relpath(thumbs[i], RAW).replace("\\", "/")
        if i in tag:
            rec["keywords"] = parse_tags(tag[i])
        if i in lic:
            rec.update(parse_license(lic[i]))
        return rec

    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / ("metadata_test.jsonl" if args.limit else "metadata.jsonl")
    temporary_file = out_file.with_name(out_file.name + ".tmp")
    n_kw = n_url = n_thumb = 0
    with open(temporary_file, "w", encoding="utf-8") as fh, ThreadPoolExecutor(args.threads) as pool:
        for k, rec in enumerate(pool.map(build, ids, chunksize=256)):
            n_kw += bool(rec["keywords"])
            n_url += rec["photo_url"] is not None
            n_thumb += rec["thumb"] is not None
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k and k % 100000 == 0:
                print(f"  {k}/{len(ids)} ({time.time() - t0:.0f}s)")
    os.replace(temporary_file, out_file)
    n = len(ids)
    print(f"Saved {out_file} ({n} records, {time.time() - t0:.0f}s)")
    print(f"with keywords: {n_kw} ({100 * n_kw / n:.1f}%) | with photo_url: {n_url} ({100 * n_url / n:.1f}%) | with thumb: {n_thumb}")


if __name__ == "__main__":
    main()