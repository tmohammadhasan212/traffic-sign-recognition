"""Download only the three official GTSRB archives and extract safely."""

import hashlib
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
ARCHIVES = ("GTSRB_Final_Training_Images.zip", "GTSRB_Final_Test_Images.zip",
            "GTSRB_Final_Test_GT.zip")


def _fetch(url: str, archive: Path) -> None:
    for attempt in range(5):
        start = archive.stat().st_size if archive.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resume = start and response.status == 206
                with archive.open("ab" if resume else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            with zipfile.ZipFile(archive) as zip_file:
                if zip_file.testzip() is not None:
                    raise ValueError(f"CRC mismatch in {archive}")
            return
        except (OSError, zipfile.BadZipFile, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def download(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    provenance = {"source": BASE, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                  "archives": {}}
    for name in ARCHIVES:
        archive = root / name
        if not archive.is_file() or not zipfile.is_zipfile(archive):
            _fetch(BASE + name, archive)
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        provenance["archives"][name] = digest.hexdigest()
        with zipfile.ZipFile(archive) as zip_file:
            for member in zip_file.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            zip_file.extractall(root)
    (root / "download_provenance.json").write_text(json.dumps(provenance, indent=2))
