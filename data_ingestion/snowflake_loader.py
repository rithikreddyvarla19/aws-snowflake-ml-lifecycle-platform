"""Snowflake raw-zone loader helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SnowflakeTableTarget:
    database: str
    schema: str
    table: str
    stage: str
    file_format: str = "TYPE = JSON STRIP_OUTER_ARRAY = FALSE"

    @property
    def fq_table(self) -> str:
        return f"{self.database}.{self.schema}.{self.table}"

    @property
    def fq_stage(self) -> str:
        return f"{self.database}.{self.schema}.{self.stage}"


class SnowflakeRawLoader:
    """Generates and optionally executes Snowflake load statements."""

    def __init__(self, connection: object | None = None) -> None:
        self.connection = connection

    def build_copy_sql(self, target: SnowflakeTableTarget, pattern: str = ".*[.]jsonl") -> list[str]:
        return [
            f"CREATE SCHEMA IF NOT EXISTS {target.database}.{target.schema};",
            f"CREATE STAGE IF NOT EXISTS {target.fq_stage};",
            (
                f"CREATE TABLE IF NOT EXISTS {target.fq_table} "
                "(raw VARIANT, source_file STRING, loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP());"
            ),
            (
                f"COPY INTO {target.fq_table} (raw, source_file) "
                f"FROM (SELECT $1, METADATA$FILENAME FROM @{target.fq_stage}) "
                f"FILE_FORMAT = ({target.file_format}) PATTERN = '{pattern}';"
            ),
        ]

    def execute_copy(self, target: SnowflakeTableTarget) -> None:
        if self.connection is None:
            raise RuntimeError("A Snowflake connection is required to execute COPY statements.")

        cursor = self.connection.cursor()
        for sql in self.build_copy_sql(target):
            cursor.execute(sql)

    @staticmethod
    def put_statement(local_file: Path, target: SnowflakeTableTarget) -> str:
        return f"PUT file://{local_file.as_posix()} @{target.fq_stage} AUTO_COMPRESS=TRUE OVERWRITE=FALSE;"

