"""Composable ingestion connectors for the raw data zone."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data_ingestion.schema import DataSource, IngestionResult, SourceType

Record = dict[str, Any]


class IngestionError(RuntimeError):
    """Raised when a source cannot be extracted."""


def _sha256_records(records: Iterable[Record]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def read_local_csv(path: Path) -> list[Record]:
    if not path.exists():
        raise IngestionError(f"CSV source does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> list[Record]:
    if not path.exists():
        raise IngestionError(f"JSONL source does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


class MultiSourceIngestor:
    """Extracts configured sources and lands immutable JSONL snapshots."""

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = raw_root
        self.raw_root.mkdir(parents=True, exist_ok=True)

    def extract(self, source: DataSource) -> list[Record]:
        if source.source_type == SourceType.LOCAL_CSV:
            return read_local_csv(source.local_path)
        if source.source_type == SourceType.JSONL:
            return read_jsonl(source.local_path)
        if source.source_type == SourceType.S3_CSV:
            return self._read_s3_csv(source)
        if source.source_type == SourceType.SNOWFLAKE:
            return self._read_snowflake(source)

        raise IngestionError(f"Unsupported source type: {source.source_type}")

    def ingest(self, source: DataSource) -> IngestionResult:
        records = self.extract(source)
        loaded_at = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        destination = self.raw_root / source.name / f"loaded_at={loaded_at}" / "part-00000.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, sort_keys=True, default=str) + "\n")

        return IngestionResult(
            source_name=source.name,
            records=len(records),
            destination_uri=str(destination),
            checksum=_sha256_records(records),
            loaded_at_utc=loaded_at,
        )

    def _read_s3_csv(self, source: DataSource) -> list[Record]:
        try:
            import boto3
        except ImportError as exc:
            raise IngestionError("Install boto3 to ingest S3 CSV sources.") from exc

        if not source.uri.startswith("s3://"):
            raise IngestionError(f"S3 sources must use s3:// URIs, got {source.uri}")

        bucket_key = source.uri.removeprefix("s3://")
        bucket, key = bucket_key.split("/", 1)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read().decode(source.options.get("encoding", "utf-8")).splitlines()
        return list(csv.DictReader(body))

    def _read_snowflake(self, source: DataSource) -> list[Record]:
        try:
            import snowflake.connector
        except ImportError as exc:
            raise IngestionError("Install snowflake-connector-python to ingest Snowflake sources.") from exc

        connection = snowflake.connector.connect(**source.options["connection"])
        try:
            sql = source.query or f"SELECT * FROM {source.table}"
            cursor = connection.cursor()
            cursor.execute(sql)
            columns = [col[0].lower() for col in cursor.description]
            return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
        finally:
            connection.close()

