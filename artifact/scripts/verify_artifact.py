#!/usr/bin/env python3
"""Verify hashes and sizes recorded in artifact/data/data_manifest.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_path = DATA / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for record in manifest["files"]:
        path = DATA / record["name"]
        if not path.is_file():
            failures.append(f"missing: {record['name']}")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"size mismatch: {record['name']}")
        if sha256(path) != record["sha256"]:
            failures.append(f"hash mismatch: {record['name']}")
    if failures:
        raise SystemExit("ARTIFACT_VERIFY=FAIL\n" + "\n".join(failures))
    print(f"ARTIFACT_VERIFY=PASS files={len(manifest['files'])}")


if __name__ == "__main__":
    main()

