"""Typed ingestion metadata shared across local, S3, and Snowflake sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SourceType(str, Enum):
    """Supported source systems."""

    LOCAL_CSV = "local_csv"
    S3_CSV = "s3_csv"
    SNOWFLAKE = "snowflake"
    JSONL = "jsonl"


@dataclass(frozen=True)
class DataSource:
    """Declarative data source definition used by ingestion jobs."""

    name: str
    source_type: SourceType
    uri: str
    primary_key: str | None = None
    table: str | None = None
    query: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> DataSource:
        return cls(
            name=name,
            source_type=SourceType(config["source_type"]),
            uri=config["uri"],
            primary_key=config.get("primary_key"),
            table=config.get("table"),
            query=config.get("query"),
            options=config.get("options", {}),
        )

    @property
    def local_path(self) -> Path:
        return Path(self.uri)


@dataclass(frozen=True)
class IngestionResult:
    """Metadata emitted after a source is extracted into the raw zone."""

    source_name: str
    records: int
    destination_uri: str
    checksum: str
    loaded_at_utc: str

