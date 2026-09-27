#!/usr/bin/env python
"""Download the official LVK sample alert notices for manual testing.

The IGWN public alerts user guide publishes a fixed set of example notices
for the mock superevent MS181101ab -- the same fixtures its own tutorial
uses -- in both the JSON (GCN) and Avro (SCIMMA) encodings:
https://emfollow.docs.ligo.org/userguide/tutorial/receiving/gcn.html

They are not committed to the repository (each JSON file is about 1 MB,
almost all of it the base64-encoded skymap); this script fetches them into
tests/data/samples so `decam-chatterbox replay` can be exercised against a
realistic, real-schema notice without needing live SCIMMA access:

    python scripts/fetch_samples.py
    decam-chatterbox replay tests/data/samples/MS181101ab-preliminary.json \\
        --dry-run
"""

import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://emfollow.docs.ligo.org/userguide/_static/"
ALERT_TYPES = ("earlywarning", "preliminary", "initial", "update", "retraction", "ext-update")
SUPEREVENT = "MS181101ab"

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "samples"


def fetch(out_dir: Path, fmt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "json" if fmt == "json" else "avro"
    for alert_type in ALERT_TYPES:
        name = f"{SUPEREVENT}-{alert_type}.{suffix}"
        url = BASE_URL + name
        dest = out_dir / name
        print(f"Fetching {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)  # fixed, documented https URL
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Where to save the files")
    parser.add_argument(
        "--format",
        choices=("json", "avro", "both"),
        default="both",
        help="json (GCN-style, base64 skymap) or avro (SCIMMA-style, raw-byte skymap)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    formats = ("json", "avro") if args.format == "both" else (args.format,)
    for fmt in formats:
        fetch(out_dir, fmt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
