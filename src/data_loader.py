"""Dataset loading utilities for the entity-resolution pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
	return pd.read_csv(
		path,
		sep="\t",
		dtype={column: "string" for column in SOURCE_COLUMNS},
		keep_default_na=True,
		nrows=nrows,
	)


def load_sources(directory: str | Path, prefix: str, nrows: int | None = None) -> dict[str, pd.DataFrame]:
	directory = Path(directory)
	return {
		f"source{number}": read_tsv(directory / f"{prefix}_source{number}.tsv", nrows=nrows)
		for number in (1, 2, 3)
	}


def load_reference_sources(directory: str | Path, prefix: str, nrows: int | None = None) -> pd.DataFrame:
	sources = load_sources(directory, prefix, nrows=nrows)
	reference = pd.concat(
		[sources["source2"].assign(record_source="source2"), sources["source3"].assign(record_source="source3")],
		ignore_index=True,
	)
	return reference


def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
	frame = pd.read_csv(
		path,
		sep="\t",
		dtype={"source1_entity_id": "string", "matched_entity_ids": "string"},
		keep_default_na=False,
	)
	return {
		str(row.source1_entity_id): {
			entity_id.strip()
			for entity_id in str(row.matched_entity_ids).split(",")
			if entity_id.strip()
		}
		for row in frame.itertuples(index=False)
	}


def summarize_frame(frame: pd.DataFrame) -> dict[str, object]:
	return {
		"rows": len(frame),
		"columns": list(frame.columns),
		"dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
		"missing": frame.isna().sum().to_dict(),
		"countries": frame["country"].value_counts(dropna=False).to_dict(),
	}


def inspect_dataset(dataset_root: str | Path, nrows: int | None = None) -> dict[str, object]:
	dataset_root = Path(dataset_root)
	train = load_sources(dataset_root / "train", "train", nrows=nrows)
	test = load_sources(dataset_root / "test", "test", nrows=nrows)
	return {
		"train": {name: summarize_frame(frame) for name, frame in train.items()},
		"test": {name: summarize_frame(frame) for name, frame in test.items()},
	}


def iter_tsv_chunks(path: str | Path, chunksize: int = 100_000) -> Iterable[pd.DataFrame]:
	return pd.read_csv(
		path,
		sep="\t",
		dtype={column: "string" for column in SOURCE_COLUMNS},
		keep_default_na=True,
		chunksize=chunksize,
	)
