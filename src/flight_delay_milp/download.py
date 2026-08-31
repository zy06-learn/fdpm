from __future__ import annotations

import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


def validate_bts_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host == "transtats.bts.gov" or host.endswith(".transtats.bts.gov")
    ):
        raise ValueError("Only HTTPS downloads from transtats.bts.gov are accepted.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
            if member.is_dir():
                continue
            target = destination.joinpath(*member_path.parts)
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite extracted file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(str(target))
    return extracted


def download_bts_export(url: str, output_dir: str | Path) -> dict[str, object]:
    validate_bts_url(url)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_path = destination / f"bts-delay-causes-{timestamp}.zip"
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {archive_path}")

    request = urllib.request.Request(url, headers={"User-Agent": "flight-delay-reproduction/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, archive_path.open("wb") as out:
        validate_bts_url(response.geturl())
        shutil.copyfileobj(response, out)
    if not zipfile.is_zipfile(archive_path):
        archive_path.unlink(missing_ok=True)
        raise ValueError("BTS response was not a ZIP archive.")

    extracted = _safe_extract(archive_path, destination)
    manifest = {
        "source_url": url,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "extracted": extracted,
    }
    manifest_path = destination / f"bts-delay-causes-{timestamp}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
