"""Run Phase 1 and Phase 2: download, prepare, train, evaluate, and report."""

from __future__ import annotations

import logging
import sys

import pandas as pd

from src.data.download import download_all
from src.data.preprocess import prepare_all
from src.models.evaluate import plot_dataset_overview, verify_saved_metrics
from src.models.train_baseline import train_all
from src.reporting.build_reports import build_reports
from src.utils.config import append_event, load_config, path_of
from src.utils.reproducibility import set_seed


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(path_of(load_config(), "logs_dir") / "pipeline.log"),
        ],
    )
    config = load_config()
    set_seed(int(config["seed"]))
    append_event(config, "pipeline", "Phase 1 and Phase 2 started")
    download_all(config)
    prepared = prepare_all(config)
    processed = path_of(config, "processed_dir")
    plot_dataset_overview(
        path_of(config, "results_dir"),
        pd.read_parquet(processed / "phiusiil.parquet"),
        pd.read_parquet(processed / "phreshphish.parquet"),
    )
    summary = train_all(config)
    verification = verify_saved_metrics(path_of(config, "results_dir"))
    build_reports(config)
    append_event(
        config,
        "pipeline",
        "Phase 1 and Phase 2 finished",
        models=summary["n_models_trained"],
        verification_passed=verification["passed"],
        phiusiil_rows=prepared["phiusiil"]["sample_size_actual"],
        phreshphish_rows=prepared["phreshphish"]["sample_size_actual"],
    )
    # Rebuild the log section so the final event is included.
    build_reports(config)
    if summary.get("errors") or not verification["passed"]:
        logging.getLogger(__name__).error("Pipeline completed with errors or failed verification")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
