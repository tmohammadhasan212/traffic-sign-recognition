"""Validate the official archives after extraction and build track-safe manifests."""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from traffic_signs.data.labels import NUM_CLASSES

TRACK_RE = re.compile(r"^(\d{5})_\d{5}\.ppm$", re.IGNORECASE)
TRAIN_COUNT = 39209
TEST_COUNT = 12630


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing annotations: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def _image_ok(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable image: {path}") from exc


def scan_training(root: Path, verify_images: bool = True) -> list[dict]:
    base = root / "GTSRB" / "Final_Training" / "Images"
    if not base.is_dir():
        raise FileNotFoundError(f"Expected extracted training images at {base}")
    rows = []
    seen = set()
    for class_id in range(NUM_CLASSES):
        directory = base / f"{class_id:05d}"
        for row in _read_csv(directory / f"GT-{class_id:05d}.csv"):
            name = Path(row["Filename"]).name
            match = TRACK_RE.fullmatch(name)
            if not match or name != row["Filename"]:
                raise ValueError(f"Invalid training filename: {row['Filename']}")
            if int(row["ClassId"]) != class_id:
                raise ValueError(f"Class mismatch: {name}")
            path = directory / name
            if not path.is_file() or path in seen:
                raise ValueError(f"Missing or duplicate image: {path}")
            seen.add(path)
            if verify_images:
                _image_ok(path)
            rows.append({"path": path.relative_to(root).as_posix(), "class_id": class_id,
                         "track_id": f"{class_id:05d}/{match.group(1)}"})
        disk = set(directory.glob("*.ppm"))
        if disk != {p for p in seen if p.parent == directory}:
            raise ValueError(f"Image/annotation mismatch in {directory}")
    return sorted(rows, key=lambda row: row["path"])


def scan_test(root: Path, verify_images: bool = True) -> list[dict]:
    base = root / "GTSRB" / "Final_Test" / "Images"
    rows = []
    seen = set()
    annotations = root / "GT-final_test.csv"
    if not annotations.is_file():
        annotations = base / "GT-final_test.csv"
    for row in _read_csv(annotations):
        name = row["Filename"]
        class_id = int(row["ClassId"])
        if Path(name).name != name or not 0 <= class_id < NUM_CLASSES:
            raise ValueError(f"Invalid test annotation: {row}")
        path = base / name
        if not path.is_file() or path in seen:
            raise ValueError(f"Missing or duplicate test image: {path}")
        seen.add(path)
        if verify_images:
            _image_ok(path)
        rows.append({"path": path.relative_to(root).as_posix(), "class_id": class_id})
    if set(base.glob("*.ppm")) != seen:
        raise ValueError("Test image/annotation mismatch")
    return sorted(rows, key=lambda row: row["path"])


def split_tracks(rows: list[dict], seed: int, validation_fraction: float) -> list[dict]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie between zero and one")
    tracks = {i: sorted({r["track_id"] for r in rows if r["class_id"] == i})
              for i in range(NUM_CLASSES)}
    rng = np.random.default_rng(seed)
    validation = set()
    for class_id, class_tracks in tracks.items():
        if len(class_tracks) < 2:
            raise ValueError(f"Class {class_id} needs at least two tracks")
        count = min(len(class_tracks) - 1, max(1, round(len(class_tracks) * validation_fraction)))
        validation.update(rng.permutation(class_tracks)[:count].tolist())
    result = [{**r, "split": "validation" if r["track_id"] in validation else "train"}
              for r in rows]
    assert not ({r["track_id"] for r in result if r["split"] == "train"} & validation)
    return result


def prepare(root: Path, output: Path, seed: int = 2026, validation_fraction: float = 0.2,
            strict_counts: bool = True, verify_images: bool = True) -> dict:
    train = scan_training(root, verify_images)
    test = scan_test(root, verify_images)
    if strict_counts and (len(train), len(test)) != (TRAIN_COUNT, TEST_COUNT):
        raise ValueError(f"Unexpected image counts: train={len(train)}, test={len(test)}")
    if set(Counter(r["class_id"] for r in train)) != set(range(NUM_CLASSES)):
        raise ValueError("Training data lacks one or more of the 43 classes")
    train = split_tracks(train, seed, validation_fraction)
    digest = hashlib.sha256("\n".join(r["path"] for r in train + test).encode()).hexdigest()
    manifest = {"schema": 1, "seed": seed, "validation_fraction": validation_fraction,
                "path_sha256": digest, "training": train, "test": test}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_or_verify_track_manifest(manifest: dict, path: Path) -> None:
    """Commit this compact split artifact; reject a future incompatible archive or split."""
    assignments = {row["track_id"]: row["split"] for row in manifest["training"]}
    document = {"seed": manifest["seed"],
                "validation_fraction": manifest["validation_fraction"],
                "path_sha256": manifest["path_sha256"],
                "tracks": dict(sorted(assignments.items()))}
    if path.exists():
        if json.loads(path.read_text()) != document:
            raise ValueError(f"Committed track split differs from prepared dataset: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n")
