"""Vectorized DuckDB-backed candidate generation for large reference sources."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .preprocessing import add_normalized_columns


class DuckDBReferenceBlocker:
    def __init__(self, database_path: str | Path, max_candidates_per_entity: int = 500) -> None:
        self.database_path = str(database_path)
        self.max_candidates_per_entity = max_candidates_per_entity
        self.connection = duckdb.connect(self.database_path)
        self.reference_count = 0

    def build(self, source2_path: str | Path, source3_path: str | Path) -> "DuckDBReferenceBlocker":
        source2 = str(Path(source2_path).resolve()).replace("'", "''")
        source3 = str(Path(source3_path).resolve()).replace("'", "''")
        self.connection.execute(
            f"""
            CREATE OR REPLACE TABLE reference AS
            SELECT *,
                lower(strip_accents(regexp_replace(coalesce(cast(business_name AS VARCHAR), ''), '[^[:alnum:][:space:]]+', ' ', 'g'))) AS name_normalized,
                lower(strip_accents(regexp_replace(coalesce(cast(business_address AS VARCHAR), ''), '[^[:alnum:][:space:]]+', ' ', 'g'))) AS address_normalized,
                lower(trim(coalesce(cast(country AS VARCHAR), ''))) AS country_normalized
            FROM (
                SELECT *, 'source2' AS record_source FROM read_csv('{source2}', delim='\\t', header=true, null_padding=true)
                UNION ALL
                SELECT *, 'source3' AS record_source FROM read_csv('{source3}', delim='\\t', header=true, null_padding=true)
            ) raw
            """
        )
        self.connection.execute(
            """
            ALTER TABLE reference ADD COLUMN house_number VARCHAR;
            ALTER TABLE reference ADD COLUMN postal_code VARCHAR;
            UPDATE reference SET
                house_number = regexp_extract(address_normalized, '(^| )([0-9]+[a-z]?)', 2),
                postal_code = regexp_extract(address_normalized, '([0-9]{4,10})', 1);
            CREATE OR REPLACE TABLE name_tokens AS
                SELECT country_normalized, unnest(string_split(name_normalized, ' ')) AS token, entity_id
                FROM reference WHERE name_normalized <> '';
            CREATE OR REPLACE TABLE address_tokens AS
                SELECT country_normalized, unnest(string_split(address_normalized, ' ')) AS token, entity_id
                FROM reference WHERE address_normalized <> '';
            """
        )
        self.reference_count = int(self.connection.execute("SELECT count(*) FROM reference").fetchone()[0])
        return self

    def generate(self, source1: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        queries = add_normalized_columns(source1)
        self.connection.register("source1_query", queries)
        self.connection.execute("DROP TABLE IF EXISTS candidate_result")
        self.connection.execute(
            f"""
            CREATE TEMP TABLE candidate_result AS
            WITH raw_candidates AS (
                SELECT q.entity_id AS source1_entity_id, r.entity_id AS candidate_entity_id, 1 AS priority
                FROM source1_query q JOIN reference r USING (country_normalized)
                WHERE (q.name_normalized <> '' AND q.name_normalized = r.name_normalized)
                   OR (q.address_normalized <> '' AND q.address_normalized = r.address_normalized)
                   OR (q.house_number <> '' AND q.house_number = r.house_number)
                   OR (q.postal_code <> '' AND q.postal_code = r.postal_code)
                UNION
                SELECT q.entity_id, t.entity_id, 2 AS priority
                FROM source1_query q JOIN name_tokens t
                  ON q.country_normalized = t.country_normalized
                 AND list_contains(string_split(q.name_normalized, ' '), t.token)
                WHERE q.name_normalized <> ''
                UNION
                SELECT q.entity_id, t.entity_id, 3 AS priority
                FROM source1_query q JOIN address_tokens t
                  ON q.country_normalized = t.country_normalized
                 AND list_contains(string_split(q.address_normalized, ' '), t.token)
                WHERE q.address_normalized <> ''
            ),
            deduplicated AS (
                SELECT source1_entity_id, candidate_entity_id, min(priority) AS priority
                FROM raw_candidates
                GROUP BY source1_entity_id, candidate_entity_id
            ),
            ranked AS (
                SELECT source1_entity_id, candidate_entity_id,
                    row_number() OVER (PARTITION BY source1_entity_id ORDER BY priority, candidate_entity_id) AS rank
                FROM deduplicated
            )
            SELECT source1_entity_id, candidate_entity_id
            FROM ranked WHERE rank <= {self.max_candidates_per_entity}
            """
        )
        pairs = self.connection.execute(
            "SELECT source1_entity_id, candidate_entity_id FROM candidate_result ORDER BY source1_entity_id, candidate_entity_id"
        ).df()
        counts = pairs.groupby("source1_entity_id").size() if not pairs.empty else pd.Series(dtype=float)
        stats = {
            "candidate_recall": float("nan"),
            "average_candidates": float(counts.mean()) if not counts.empty else 0.0,
            "median_candidates": float(counts.median()) if not counts.empty else 0.0,
            "maximum_candidates": float(counts.max()) if not counts.empty else 0.0,
            "candidate_reduction_ratio": 1.0 - len(pairs) / (len(source1) * self.reference_count) if len(source1) and self.reference_count else 1.0,
        }
        return pairs, stats

    def fetch_records(self, entity_ids: list[str]) -> pd.DataFrame:
        if not entity_ids:
            return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country", "record_source"])
        self.connection.register("requested_ids", pd.DataFrame({"entity_id": entity_ids}))
        return self.connection.execute(
            "SELECT r.entity_id, r.business_name, r.business_address, r.country, r.record_source FROM reference r JOIN requested_ids q USING (entity_id)"
        ).df()

    def close(self) -> None:
        self.connection.close()
