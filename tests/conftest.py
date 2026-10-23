import csv
from pathlib import Path

import pytest
from PIL import Image

from traffic_signs.data.prepare import prepare


@pytest.fixture
def tiny_gtsrb(tmp_path: Path):
    """A structurally complete 43-class fixture with two independent tracks per class."""
    training = tmp_path / "GTSRB" / "Final_Training" / "Images"
    for class_id in range(43):
        directory = training / f"{class_id:05d}"
        directory.mkdir(parents=True)
        with (directory / f"GT-{class_id:05d}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["Filename", "Width", "Height", "Roi.X1", "Roi.Y1",
                             "Roi.X2", "Roi.Y2", "ClassId"])
            for track in range(2):
                name = f"{track:05d}_00000.ppm"
                Image.new("RGB", (32, 24), (class_id * 5 % 255, track * 100, 30)).save(
                    directory / name)
                writer.writerow([name, 32, 24, 0, 0, 31, 23, class_id])
    test = tmp_path / "GTSRB" / "Final_Test" / "Images"
    test.mkdir(parents=True)
    with (test / "GT-final_test.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["Filename", "Width", "Height", "Roi.X1", "Roi.Y1",
                         "Roi.X2", "Roi.Y2", "ClassId"])
        for class_id in range(43):
            name = f"{class_id:05d}.ppm"
            Image.new("RGB", (30, 34), (class_id * 5 % 255, 0, 30)).save(test / name)
            writer.writerow([name, 30, 34, 0, 0, 29, 33, class_id])
    manifest_path = tmp_path / "manifest.json"
    manifest = prepare(tmp_path, manifest_path, strict_counts=False)
    return tmp_path, manifest_path, manifest
