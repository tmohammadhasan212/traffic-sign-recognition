"""Download only the three official GTSRB archives and extract safely."""

import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
ARCHIVES = ("GTSRB_Final_Training_Images.zip", "GTSRB_Final_Test_Images.zip",
            "GTSRB_Final_Test_GT.zip")


def download(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    provenance = {"source": BASE, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                  "archives": {}}
    for name in ARCHIVES:
        archive = root / name
        if not archive.is_file():
            urllib.request.urlretrieve(BASE + name, archive)
        provenance["archives"][name] = hashlib.sha256(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as zip_file:
            for member in zip_file.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            zip_file.extractall(root)
    (root / "download_provenance.json").write_text(json.dumps(provenance, indent=2))
