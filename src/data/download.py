"""Retrieve PhiUSIIL and the URL/label/date columns of PhreshPhish.

The original PhiUSIIL zip is left unchanged. PhreshPhish HTML is never read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from src.utils.config import append_event, load_config, path_of

LOGGER = logging.getLogger(__name__)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_phiusiil(config: dict) -> dict:
    raw_dir = path_of(config, "raw_dir") / "phiusiil"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "phiusiil_phishing_url_dataset.zip"
    extract_dir = raw_dir / "extracted"
    csv_path = extract_dir / "PhiUSIIL_Phishing_URL_Dataset.csv"
    if not zip_path.exists():
        import urllib.request

        LOGGER.info("Downloading PhiUSIIL zip")
        urllib.request.urlretrieve(config["phiusiil"]["zip_url"], zip_path)
    if not csv_path.exists():
        LOGGER.info("Extracting PhiUSIIL CSV")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    info = {
        "dataset": "phiusiil",
        "source_url": config["phiusiil"]["zip_url"],
        "uci_page": config["phiusiil"]["uci_page"],
        "download_date": datetime.now(timezone.utc).date().isoformat(),
        "zip_path": str(zip_path.relative_to(config["_root"])),
        "csv_path": str(csv_path.relative_to(config["_root"])),
        "zip_sha256": _sha256(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "csv_bytes": csv_path.stat().st_size,
        "note": "Zip archive stored unchanged. CSV is the extracted original file.",
    }
    meta = path_of(config, "metadata_dir") / "phiusiil_download.json"
    meta.write_text(json.dumps(info, indent=2))
    append_event(config, "download", "PhiUSIIL archive present", sha256=info["zip_sha256"])
    return info


def _list_parquet_files(repo_id: str) -> list[str]:
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    selected = [name for name in files if name.startswith("data/") and name.endswith(".parquet")]
    return sorted(selected)


def download_phreshphish(config: dict) -> dict:
    """Read url, label, date, and sha256 from every parquet shard.

    Column projection avoids materialising the HTML field. Each shard is
    checkpointed so a rerun does not read it again.
    """
    spec = config["phreshphish"]
    interim = path_of(config, "interim_dir") / "phreshphish_columns"
    interim.mkdir(parents=True, exist_ok=True)
    combined_path = path_of(config, "interim_dir") / "phreshphish_columns.parquet"
    manifest_path = path_of(config, "metadata_dir") / "phreshphish_download.json"
    if combined_path.exists() and manifest_path.exists():
        LOGGER.info("Using cached PhreshPhish column extract")
        return json.loads(manifest_path.read_text())

    files = _list_parquet_files(spec["repo_id"])
    filesystem = HfFileSystem()
    columns = spec["columns"]

    def _process_shard(name: str):
        checkpoint = interim / (name.replace("/", "__") + ".parquet")
        if checkpoint.exists():
            frame = pd.read_parquet(checkpoint)
        else:
            remote = f"datasets/{spec['repo_id']}/{name}"
            with filesystem.open(remote, "rb") as handle:
                parquet_file = pq.ParquetFile(handle)
                available = list(parquet_file.schema_arrow.names)
                missing = [column for column in columns if column not in available]
                if missing:
                    raise RuntimeError(f"{name} is missing {missing}. Available columns: {available}")
                table = parquet_file.read(columns=columns)
            frame = table.to_pandas()
            frame["source_file"] = name
            frame["hf_split"] = "train" if "/train-" in f"/{name}" or name.startswith("data/train-") else "test"
            frame.to_parquet(checkpoint, index=False)
        stat = {
            "file": name,
            "rows": int(len(frame)),
            "hf_split": str(frame["hf_split"].iloc[0]) if len(frame) else None,
        }
        return name, frame, stat

    frames_by_name = {}
    stats_by_name = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(_process_shard, name): name for name in files}
        for future in as_completed(future_map):
            name, frame, stat = future.result()
            frames_by_name[name] = frame
            stats_by_name[name] = stat
            LOGGER.info("Completed PhreshPhish shard %s rows=%s", name, len(frame))

    frames = [frames_by_name[name] for name in files]
    shard_stats = [stats_by_name[name] for name in files]

    if not frames:
        raise RuntimeError("No PhreshPhish parquet shards were found.")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(combined_path, index=False)
    info = {
        "dataset": "phreshphish",
        "repo_id": spec["repo_id"],
        "hf_page": spec["hf_page"],
        "paper_url": spec["paper_url"],
        "download_date": datetime.now(timezone.utc).date().isoformat(),
        "columns_read": columns + ["source_file", "hf_split"],
        "columns_explicitly_not_read": ["html", "target", "lang", "lang_score"],
        "n_shards": len(files),
        "n_rows_retrieved": int(len(combined)),
        "shard_stats": shard_stats,
        "interim_path": str(combined_path.relative_to(config["_root"])),
        "version_note": "Dataset card v1.0.1 (2026-02-07) is the repository version inspected.",
    }
    manifest_path.write_text(json.dumps(info, indent=2))
    append_event(config, "download", "PhreshPhish columns retrieved", rows=int(len(combined)), shards=len(files))
    return info


def download_all(config: dict | None = None) -> dict:
    config = config or load_config()
    return {"phiusiil": download_phiusiil(config), "phreshphish": download_phreshphish(config)}
