"""CLI entry point for landing raw customer churn source snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from data_ingestion.connectors import MultiSourceIngestor
from data_ingestion.schema import DataSource


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_ingestion(config_path: Path, raw_root: Path, source_names: list[str] | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    configured_sources = config["data_sources"]
    selected = source_names or list(configured_sources)
    ingestor = MultiSourceIngestor(raw_root=raw_root)
    results = []

    for source_name in selected:
        source = DataSource.from_config(source_name, configured_sources[source_name])
        result = ingestor.ingest(source)
        results.append(result.__dict__)

    manifest = raw_root / "ingestion_manifest.json"
    manifest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Land raw churn data snapshots.")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", action="append", dest="sources", help="Source name to ingest. Repeatable.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run_ingestion(args.config, args.raw_root, args.sources), indent=2))

