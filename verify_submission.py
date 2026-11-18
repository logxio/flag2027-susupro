"""Check a FLAG 2027 submission ZIP against fixed official trial IDs."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from contextlib import nullcontext
from pathlib import Path


EXPECTED = {
    "no_gender/sub_score_v4_English_heard.txt": ("no_gender", "English", 1008),
    "no_gender/sub_score_v4_Bangla_unheard.txt": ("no_gender", "Bangla", 1406),
    "gender/sub_score_v4_English_heard.txt": ("gender", "English", 982),
    "gender/sub_score_v4_Bangla_unheard.txt": ("gender", "Bangla", 1468),
}


def official_ids(archive: zipfile.ZipFile, track: str, language: str) -> list[str]:
    suffix = f"/{track}/{language}_test.txt"
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"Expected one official list ending {suffix}, found {names}")
    lines = archive.read(names[0]).decode().splitlines()
    return [line.split()[0] for line in lines if line.strip()]


def verify(submission: Path, dev_zip: Path | None = None, pair_ids_json: Path | None = None,
           manifest: Path | None = None) -> dict[str, int]:
    if sum(value is not None for value in (dev_zip, pair_ids_json, manifest)) != 1:
        raise ValueError("Give exactly one of --dev-zip, --pair-ids-json or --manifest")
    ids_manifest = json.loads(pair_ids_json.read_text()) if pair_ids_json else None
    expected = EXPECTED
    if manifest is not None:
        metadata = json.loads(manifest.read_text())
        ids_manifest = metadata["pair_ids"]
        expected = {cell["relative"]: (None, None, cell["rows"])
                    for cell in metadata["cells"].values()}
        if len(expected) != 4 or set(expected) != set(EXPECTED):
            raise ValueError("Manifest must describe the required four score files")
    report = {}
    with zipfile.ZipFile(submission) as package:
        members = package.namelist()
        if len(members) != 4 or set(members) != set(EXPECTED):
            raise ValueError(f"Submission members differ from required four: {members}")
        with zipfile.ZipFile(dev_zip) if dev_zip else nullcontext() as official:
            for path, (track, language, expected_rows) in expected.items():
                reference = official_ids(official, track, language) if official else ids_manifest[path]
                if len(reference) != expected_rows or len(set(reference)) != expected_rows:
                    raise ValueError(f"Official pair list invalid: {path}")
                lines = package.read(path).decode().splitlines()
                parsed = []
                for index, line in enumerate(lines, 1):
                    fields = line.split()
                    if len(fields) != 2:
                        raise ValueError(f"{path}:{index}: expected ID and distance")
                    value = float(fields[1])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"{path}:{index}: invalid distance {fields[1]}")
                    parsed.append(fields[0])
                if parsed != reference:
                    raise ValueError(f"{path}: submitted IDs differ from the official trial pairs")
                report[path] = len(parsed)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--dev-zip", type=Path)
    parser.add_argument("--pair-ids-json", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest
    if not any((args.dev_zip, args.pair_ids_json, manifest)):
        manifest = Path(__file__).resolve().with_name("manifest.json")
    print(json.dumps(verify(args.submission, args.dev_zip, args.pair_ids_json, manifest), indent=2))
