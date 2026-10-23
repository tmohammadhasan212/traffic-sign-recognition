import json

import pytest
from PIL import Image

from traffic_signs.data.dataset import ManifestDataset, transform
from traffic_signs.data.labels import CLASS_NAMES
from traffic_signs.data.prepare import prepare, write_or_verify_track_manifest


def test_mapping_and_track_disjoint_manifest(tiny_gtsrb):
    root, path, manifest = tiny_gtsrb
    assert len(CLASS_NAMES) == 43
    assert len(manifest["training"]) == 86
    train = {row["track_id"] for row in manifest["training"] if row["split"] == "train"}
    validation = {row["track_id"] for row in manifest["training"] if row["split"] == "validation"}
    assert not train & validation
    assert len(train) == len(validation) == 43
    assert prepare(root, path, strict_counts=False) == manifest


def test_dataset_alignment_and_transforms(tiny_gtsrb):
    root, path, _ = tiny_gtsrb
    data = ManifestDataset(root, path, "validation", "cnn")
    image, label = data[0]
    assert image.shape == (3, 64, 64) and label == data.rows[0]["class_id"]
    assert transform("resnet18")(Image.new("RGB", (30, 20))).shape == (3, 224, 224)
    subset = ManifestDataset(root, path, "train", "cnn", limit=32)
    assert len({row["class_id"] for row in subset.rows}) == 32


def test_rejects_class_mismatch(tiny_gtsrb):
    root, path, _ = tiny_gtsrb
    csv_path = root / "GTSRB/Final_Training/Images/00000/GT-00000.csv"
    csv_path.write_text(csv_path.read_text().replace(";0\n", ";1\n", 1))
    with pytest.raises(ValueError, match="Class mismatch"):
        prepare(root, path, strict_counts=False)


def test_manifest_is_json(tiny_gtsrb):
    _, path, manifest = tiny_gtsrb
    assert json.loads(path.read_text())["path_sha256"] == manifest["path_sha256"]


def test_committed_track_assignments_detect_drift(tiny_gtsrb, tmp_path):
    _, _, manifest = tiny_gtsrb
    path = tmp_path / "tracks.json"
    write_or_verify_track_manifest(manifest, path)
    write_or_verify_track_manifest(manifest, path)
    changed = {**manifest, "seed": 999}
    with pytest.raises(ValueError, match="differs"):
        write_or_verify_track_manifest(changed, path)
