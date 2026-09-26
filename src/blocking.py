"""Candidate generation using memory-conscious inverted indexes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import sqlite3
from typing import DefaultDict, Iterable

import pandas as pd

from .preprocessing import add_normalized_columns


def _character_ngrams(value: str, sizes: tuple[int, ...]) -> set[str]:
	padded = f"  {value}  "
	return {
		padded[start : start + size]
		for size in sizes
		for start in range(max(0, len(padded) - size + 1))
	}


@dataclass(frozen=True)
class BlockingConfig:
	name_token_limit: int = 100
	address_token_limit: int = 100
	numeric_limit: int = 100
	character_ngram_limit: int = 100
	max_candidates_per_entity: int = 500
	character_ngram_sizes: tuple[int, ...] = (3, 4, 5)
	use_character_ngrams: bool = True
	use_token_indexes: bool = True


class CandidateBlocker:
	def __init__(self, config: BlockingConfig | None = None) -> None:
		self.config = config or BlockingConfig()
		self.reference: pd.DataFrame | None = None
		self._name_exact: DefaultDict[tuple[str, str], list[str]] = defaultdict(list)
		self._address_exact: DefaultDict[tuple[str, str], list[str]] = defaultdict(list)
		self._name_address_exact: DefaultDict[tuple[str, str, str], list[str]] = defaultdict(list)
		self._house_numbers: DefaultDict[tuple[str, str], list[str]] = defaultdict(list)
		self._postal_codes: DefaultDict[tuple[str, str], list[str]] = defaultdict(list)
		self._name_tokens: DefaultDict[tuple[str, str], list[str]] = defaultdict(list)
		self._address_tokens: DefaultDict[tuple[str, str], list[str]] = defaultdict(list)
		self._name_ngrams: DefaultDict[tuple[str, str], list[str]] = defaultdict(list)
		self._address_ngrams: DefaultDict[tuple[str, str], list[str]] = defaultdict(list)

	def fit(self, reference: pd.DataFrame) -> "CandidateBlocker":
		self.reference = add_normalized_columns(reference).reset_index(drop=True)
		for row in self.reference.itertuples(index=False):
			entity_id = str(row.entity_id)
			country = str(row.country_normalized)
			if row.name_normalized:
				self._name_exact[(country, row.name_normalized)].append(entity_id)
				for token in set(row.name_normalized.split()):
					self._name_tokens[(country, token)].append(entity_id)
				if self.config.use_character_ngrams:
					for gram in _character_ngrams(row.name_normalized, self.config.character_ngram_sizes):
						self._name_ngrams[(country, gram)].append(entity_id)
			if row.address_normalized:
				self._address_exact[(country, row.address_normalized)].append(entity_id)
				for token in set(row.address_normalized.split()):
					self._address_tokens[(country, token)].append(entity_id)
				if self.config.use_character_ngrams:
					for gram in _character_ngrams(row.address_normalized, self.config.character_ngram_sizes):
						self._address_ngrams[(country, gram)].append(entity_id)
			if row.name_normalized and row.address_normalized:
				self._name_address_exact[(country, row.name_normalized, row.address_normalized)].append(entity_id)
			if row.house_number:
				self._house_numbers[(country, row.house_number)].append(entity_id)
			if row.postal_code:
				self._postal_codes[(country, row.postal_code)].append(entity_id)
		self._trim_postings()
		return self

	def _trim_postings(self) -> None:
		posting_limit = self.config.max_candidates_per_entity
		for index in (
			self._name_tokens,
			self._address_tokens,
			self._name_ngrams,
			self._address_ngrams,
			self._house_numbers,
			self._postal_codes,
		):
			for key, values in index.items():
				index[key] = values[:posting_limit]
		for index in (self._name_exact, self._address_exact, self._name_address_exact):
			for key, values in index.items():
				index[key] = values

	@staticmethod
	def _extend(target: set[str], values: Iterable[str], limit: int) -> None:
		for value in values:
			target.add(value)
			if len(target) >= limit:
				return

	def _candidates_for_row(self, row: object) -> set[str]:
		country = str(row.country_normalized)
		passes: list[set[str]] = []
		limit = self.config.max_candidates_per_entity
		if row.name_normalized and row.address_normalized:
			passes.append(set(self._name_address_exact[(country, row.name_normalized, row.address_normalized)]))
		if row.name_normalized:
			name_candidates: set[str] = set(self._name_exact[(country, row.name_normalized)])
			for token in set(row.name_normalized.split()):
				self._extend(name_candidates, self._name_tokens[(country, token)], self.config.name_token_limit)
			if self.config.use_character_ngrams:
				for gram in _character_ngrams(row.name_normalized, self.config.character_ngram_sizes):
					self._extend(name_candidates, self._name_ngrams[(country, gram)], self.config.character_ngram_limit)
			passes.append(name_candidates)
		if row.address_normalized:
			address_candidates: set[str] = set(self._address_exact[(country, row.address_normalized)])
			for token in set(row.address_normalized.split()):
				self._extend(address_candidates, self._address_tokens[(country, token)], self.config.address_token_limit)
			if self.config.use_character_ngrams:
				for gram in _character_ngrams(row.address_normalized, self.config.character_ngram_sizes):
					self._extend(address_candidates, self._address_ngrams[(country, gram)], self.config.character_ngram_limit)
			passes.append(address_candidates)
		if row.house_number:
			passes.append(set(self._house_numbers[(country, row.house_number)][: self.config.numeric_limit]))
		if row.postal_code:
			passes.append(set(self._postal_codes[(country, row.postal_code)][: self.config.numeric_limit]))
		candidates = set().union(*passes) if passes else set()
		if len(candidates) <= limit:
			return candidates
		return set(sorted(candidates)[:limit])

	def generate(self, source1: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
		if self.reference is None:
			raise RuntimeError("CandidateBlocker.fit must be called before generate")
		queries = add_normalized_columns(source1)
		rows: list[dict[str, str]] = []
		counts: list[int] = []
		for row in queries.itertuples(index=False):
			candidate_ids = self._candidates_for_row(row)
			counts.append(len(candidate_ids))
			rows.extend(
				{"source1_entity_id": str(row.entity_id), "candidate_entity_id": entity_id}
				for entity_id in sorted(candidate_ids)
			)
		candidate_pairs = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])
		if counts:
			count_series = pd.Series(counts)
			stats = {
				"candidate_recall": float("nan"),
				"average_candidates": float(count_series.mean()),
				"median_candidates": float(count_series.median()),
				"maximum_candidates": float(count_series.max()),
				"candidate_reduction_ratio": 1.0 - len(candidate_pairs) / (len(source1) * len(self.reference)),
			}
		else:
			stats = {
				"candidate_recall": float("nan"),
				"average_candidates": 0.0,
				"median_candidates": 0.0,
				"maximum_candidates": 0.0,
				"candidate_reduction_ratio": 1.0,
			}
		return candidate_pairs, stats


def evaluate_blocking_recall(candidate_pairs: pd.DataFrame, ground_truth: dict[str, set[str]]) -> float:
	grouped = candidate_pairs.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()
	evaluated = [truth.issubset(grouped.get(entity_id, set())) for entity_id, truth in ground_truth.items()]
	return float(sum(evaluated) / len(evaluated)) if evaluated else 1.0


class DiskReferenceBlocker:
	"""SQLite-backed blocker that avoids keeping the full reference in RAM."""

	_INDEX_TABLES = {
		"name_exact": "country_normalized, name_normalized, entity_id",
		"address_exact": "country_normalized, address_normalized, entity_id",
		"name_address_exact": "country_normalized, name_normalized, address_normalized, entity_id",
		"house_numbers": "country_normalized, house_number, entity_id",
		"postal_codes": "country_normalized, postal_code, entity_id",
		"name_tokens": "country_normalized, token, entity_id",
		"address_tokens": "country_normalized, token, entity_id",
	}

	def __init__(self, database_path: str, config: BlockingConfig | None = None) -> None:
		self.database_path = database_path
		self.config = config or BlockingConfig()
		self.connection = sqlite3.connect(database_path)
		self.reference_count = 0

	def build(self, reference_chunks: Iterable[pd.DataFrame]) -> "DiskReferenceBlocker":
		self.connection.executescript(
			"""
			DROP TABLE IF EXISTS records;
			CREATE TABLE records (
				entity_id TEXT PRIMARY KEY, business_name TEXT, business_address TEXT,
				country TEXT, record_source TEXT, name_normalized TEXT,
				address_normalized TEXT, country_normalized TEXT, house_number TEXT,
				postal_code TEXT, address_numeric_tokens TEXT, name_numeric_tokens TEXT
			);
			"""
		)
		for table, columns in self._INDEX_TABLES.items():
			self.connection.execute(f"DROP TABLE IF EXISTS {table}")
			column_definitions = ", ".join(f"{column} TEXT" for column in columns.split(", ") )
			self.connection.execute(f"CREATE TABLE {table} ({column_definitions})")
		for frame in reference_chunks:
			prepared = add_normalized_columns(frame)
			self._insert_chunk(prepared)
			self.connection.commit()
			print({"indexed_reference_rows": self.reference_count}, flush=True)
		self.connection.execute("CREATE INDEX IF NOT EXISTS records_entity_idx ON records(entity_id)")
		for table, columns in self._INDEX_TABLES.items():
			self.connection.execute(f"CREATE INDEX IF NOT EXISTS {table}_key_idx ON {table}({columns.rsplit(', ', 1)[0]})")
		self.connection.commit()
		return self

	def _insert_chunk(self, frame: pd.DataFrame) -> None:
		record_columns = [
			"entity_id", "business_name", "business_address", "country", "record_source",
			"name_normalized", "address_normalized", "country_normalized", "house_number",
			"postal_code", "address_numeric_tokens", "name_numeric_tokens",
		]
		def text(value: object) -> str:
			return "" if pd.isna(value) else str(value)
		records = [
		tuple(text(value) for value in row)
		for row in frame[record_columns].fillna("").astype(str).itertuples(index=False, name=None)
	]
		self.connection.executemany(
			f"INSERT OR REPLACE INTO records ({', '.join(record_columns)}) VALUES ({', '.join('?' for _ in record_columns)})",
			records,
		)
		self.reference_count += len(records)
		index_rows = {table: [] for table in self._INDEX_TABLES}
		for row in frame[["entity_id", "country_normalized", "name_normalized", "address_normalized", "house_number", "postal_code"]].fillna("").astype(str).itertuples(index=False, name=None):
			entity_id, country, name, address, house_number, postal_code = row
			if name:
				index_rows["name_exact"].append((country, name, entity_id))
				if self.config.use_token_indexes:
					index_rows["name_tokens"].extend((country, token, entity_id) for token in set(name.split()))
			if address:
				index_rows["address_exact"].append((country, address, entity_id))
				if self.config.use_token_indexes:
					index_rows["address_tokens"].extend((country, token, entity_id) for token in set(address.split()))
			if name and address:
				index_rows["name_address_exact"].append((country, name, address, entity_id))
			if house_number:
				index_rows["house_numbers"].append((country, house_number, entity_id))
			if postal_code:
				index_rows["postal_codes"].append((country, postal_code, entity_id))
		for table, rows in index_rows.items():
			if rows:
				columns = self._INDEX_TABLES[table].split(", ")
				self.connection.executemany(
					f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
					rows,
				)

	def _lookup(self, table: str, values: tuple[str, ...], limit: int) -> set[str]:
		columns = self._INDEX_TABLES[table].split(", ")[:-1]
		where = " AND ".join(f"{column} = ?" for column in columns)
		rows = self.connection.execute(
			f"SELECT entity_id FROM {table} WHERE {where} LIMIT ?", (*values, limit)
		).fetchall()
		return {row[0] for row in rows}

	def candidates_for_row(self, row: object) -> set[str]:
		country = str(row.country_normalized)
		passes = []
		if row.name_normalized and row.address_normalized:
			passes.append(self._lookup("name_address_exact", (country, str(row.name_normalized), str(row.address_normalized)), self.config.max_candidates_per_entity))
		if row.name_normalized:
			name = self._lookup("name_exact", (country, str(row.name_normalized)), self.config.max_candidates_per_entity)
			if self.config.use_token_indexes:
				for token in set(str(row.name_normalized).split()):
					name.update(self._lookup("name_tokens", (country, token), self.config.name_token_limit))
			passes.append(name)
		if row.address_normalized:
			address = self._lookup("address_exact", (country, str(row.address_normalized)), self.config.max_candidates_per_entity)
			if self.config.use_token_indexes:
				generic_address_tokens = {"st", "rd", "ave", "blvd", "dr", "ln", "unit", "apt", "fl", "floor", "ste", "suite", "near", "opp", "opposite", "road", "street", "avenue", "lane", "colony", "dist", "district", "city", "state", "india", "us", "france", "de", "du", "la", "le"}
				for token in set(str(row.address_normalized).split()):
					if len(token) > 2 and token not in generic_address_tokens:
						address.update(self._lookup("address_tokens", (country, token), self.config.address_token_limit))
			passes.append(address)
		if row.house_number:
			passes.append(self._lookup("house_numbers", (country, str(row.house_number)), self.config.numeric_limit))
		if row.postal_code:
			passes.append(self._lookup("postal_codes", (country, str(row.postal_code)), self.config.numeric_limit))
		candidates = set().union(*passes) if passes else set()
		return set(sorted(candidates)[: self.config.max_candidates_per_entity])

	def generate(self, source1: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
		queries = add_normalized_columns(source1)
		rows = []
		counts = []
		for row in queries.itertuples(index=False):
			candidate_ids = self.candidates_for_row(row)
			counts.append(len(candidate_ids))
			rows.extend({"source1_entity_id": str(row.entity_id), "candidate_entity_id": entity_id} for entity_id in sorted(candidate_ids))
		pairs = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])
		count_series = pd.Series(counts, dtype=float)
		stats = {
			"candidate_recall": float("nan"),
			"average_candidates": float(count_series.mean()) if counts else 0.0,
			"median_candidates": float(count_series.median()) if counts else 0.0,
			"maximum_candidates": float(count_series.max()) if counts else 0.0,
			"candidate_reduction_ratio": 1.0 - len(pairs) / (len(source1) * self.reference_count) if source1.size and self.reference_count else 1.0,
		}
		return pairs, stats

	def fetch_records(self, entity_ids: Iterable[str]) -> pd.DataFrame:
		entity_ids = list(entity_ids)
		if not entity_ids:
			return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country", "record_source"])
		rows = []
		for start in range(0, len(entity_ids), 900):
			batch = entity_ids[start : start + 900]
			placeholders = ",".join("?" for _ in batch)
			rows.extend(self.connection.execute(f"SELECT entity_id,business_name,business_address,country,record_source FROM records WHERE entity_id IN ({placeholders})", batch).fetchall())
		return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country", "record_source"])

	def close(self) -> None:
		self.connection.close()
