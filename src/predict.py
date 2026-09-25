"""Prediction and submission-file utilities."""

from __future__ import annotations

import argparse
from itertools import chain
import os
from pathlib import Path

import pandas as pd

from .blocking import BlockingConfig, CandidateBlocker, DiskReferenceBlocker
from .data_loader import iter_tsv_chunks, load_reference_sources, load_sources
from .duckdb_blocking import DuckDBReferenceBlocker
from .evaluate import predictions_at_threshold
from .features import build_pair_features
from .train import load_model, predict_probabilities


def score_candidates(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
	scored = features.copy()
	scored["match_probability"] = predict_probabilities(bundle, scored)
	return scored


def build_submission_frames(
	source1: pd.DataFrame,
	candidate_pairs: pd.DataFrame,
	scored_pairs: pd.DataFrame,
	threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	source1_ids = source1["entity_id"].astype(str).tolist()
	candidate_map = candidate_pairs.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()
	matches = predictions_at_threshold(scored_pairs, threshold)
	candidate_rows = [
		{
			"source1_entity_id": entity_id,
			"candidate_entity_ids": ",".join(sorted(candidate_map.get(entity_id, set()))),
		}
		for entity_id in source1_ids
	]
	matching_rows = [
		{
			"source1_entity_id": entity_id,
			"matched_entity_ids": ",".join(sorted(matches.get(entity_id, set()))),
		}
		for entity_id in source1_ids
	]
	return pd.DataFrame(candidate_rows), pd.DataFrame(matching_rows)


def write_submission(
	source1: pd.DataFrame,
	candidate_pairs: pd.DataFrame,
	scored_pairs: pd.DataFrame,
	output_directory: str | Path,
	threshold: float,
) -> tuple[Path, Path]:
	output_directory = Path(output_directory)
	output_directory.mkdir(parents=True, exist_ok=True)
	candidate_frame, matching_frame = build_submission_frames(source1, candidate_pairs, scored_pairs, threshold)
	candidate_path = output_directory / "candidate_pairs.tsv"
	matching_path = output_directory / "matching_results.tsv"
	candidate_frame.to_csv(candidate_path, sep="\t", index=False)
	matching_frame.to_csv(matching_path, sep="\t", index=False)
	return candidate_path, matching_path


def run_prediction(
	dataset_root: str | Path = "dataset",
	model_path: str | Path = "models/sample_baseline.joblib",
	output_directory: str | Path = "output/sample",
	sample_source1: int | None = None,
	reference_limit: int | None = None,
	threshold: float = 0.3,
) -> tuple[Path, Path]:
	dataset_root = Path(dataset_root)
	test_sources = load_sources(dataset_root / "test", "test", nrows=sample_source1)
	source1 = test_sources["source1"]
	reference = load_reference_sources(dataset_root / "test", "test", nrows=reference_limit)
	blocker = CandidateBlocker(
		BlockingConfig(
			name_token_limit=50,
			address_token_limit=50,
			character_ngram_limit=50,
			max_candidates_per_entity=500,
		)
	)
	candidate_pairs, blocking_stats = blocker.fit(reference).generate(source1)
	features = build_pair_features(source1, reference, candidate_pairs)
	bundle = load_model(model_path)
	scored_pairs = score_candidates(bundle, features)
	paths = write_submission(source1, candidate_pairs, scored_pairs, output_directory, threshold)
	print(
		{
			"source1_rows": len(source1),
			"reference_rows": len(reference),
			"candidate_rows": len(candidate_pairs),
			"blocking": blocking_stats,
			"threshold": threshold,
			"outputs": [str(path) for path in paths],
		}
	)
	return paths


def run_chunked_prediction(
	dataset_root: str | Path = "dataset",
	model_path: str | Path = "models/sample_lightgbm_2000_numeric.joblib",
	output_directory: str | Path = "output/chunked",
	reference_db: str | Path = "models/test_reference.sqlite",
	chunk_size: int = 50_000,
	threshold: float = 0.9,
	sample_source1: int | None = None,
	reference_limit: int | None = None,
) -> tuple[Path, Path]:
	dataset_root = Path(dataset_root)
	output_directory = Path(output_directory)
	output_directory.mkdir(parents=True, exist_ok=True)
	config = BlockingConfig(
		name_token_limit=50,
		address_token_limit=50,
		numeric_limit=100,
		max_candidates_per_entity=500,
		use_token_indexes=True,
	)
	requested_db = Path(reference_db)
	reference_db = requested_db.with_name(f"{requested_db.stem}_{os.getpid()}{requested_db.suffix}")
	index = DiskReferenceBlocker(str(reference_db), config)
	def limited_chunks(path: Path, limit: int | None):
		remaining = limit
		for chunk in iter_tsv_chunks(path, chunksize=chunk_size):
			if remaining is not None:
				if remaining <= 0:
					break
				chunk = chunk.head(remaining)
				remaining -= len(chunk)
			if not chunk.empty:
				yield chunk
	reference_chunks = chain(
		(chunk.assign(record_source="source2") for chunk in limited_chunks(dataset_root / "test" / "test_source2.tsv", reference_limit)),
		(chunk.assign(record_source="source3") for chunk in limited_chunks(dataset_root / "test" / "test_source3.tsv", reference_limit)),
	)
	index.build(reference_chunks)
	bundle = load_model(model_path)
	candidate_path = output_directory / "candidate_pairs.tsv"
	matching_path = output_directory / "matching_results.tsv"
	first_write = True
	for source1_chunk in limited_chunks(dataset_root / "test" / "test_source1.tsv", sample_source1):
		candidate_pairs, blocking_stats = index.generate(source1_chunk)
		reference = index.fetch_records(candidate_pairs["candidate_entity_id"].unique())
		features = build_pair_features(source1_chunk, reference, candidate_pairs)
		scored_pairs = score_candidates(bundle, features)
		candidate_map = candidate_pairs.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()
		matches = predictions_at_threshold(scored_pairs, threshold)
		candidate_frame = pd.DataFrame(
			{
				"source1_entity_id": source1_chunk["entity_id"].astype(str),
				"candidate_entity_ids": [",".join(sorted(candidate_map.get(str(entity_id), set()))) for entity_id in source1_chunk["entity_id"]],
			}
		)
		matching_frame = pd.DataFrame(
			{
				"source1_entity_id": source1_chunk["entity_id"].astype(str),
				"matched_entity_ids": [",".join(sorted(matches.get(str(entity_id), set()))) for entity_id in source1_chunk["entity_id"]],
			}
		)
		candidate_frame.to_csv(candidate_path, sep="\t", index=False, mode="w" if first_write else "a", header=first_write)
		matching_frame.to_csv(matching_path, sep="\t", index=False, mode="w" if first_write else "a", header=first_write)
		first_write = False
		print({"processed_source1": len(source1_chunk), "candidates": len(candidate_pairs), "blocking": blocking_stats})
	index.close()
	return candidate_path, matching_path


def run_duckdb_prediction(
	dataset_root: str | Path = "dataset",
	model_path: str | Path = "models/sample_lightgbm_2000_numeric.joblib",
	output_directory: str | Path = "output/full_submission",
	database_path: str | Path = "models/test_reference.duckdb",
	chunk_size: int = 50_000,
	threshold: float = 0.9,
) -> tuple[Path, Path]:
	dataset_root = Path(dataset_root)
	output_directory = Path(output_directory)
	output_directory.mkdir(parents=True, exist_ok=True)
	blocker = DuckDBReferenceBlocker(database_path, max_candidates_per_entity=500)
	blocker.build(
		dataset_root / "test" / "test_source2.tsv",
		dataset_root / "test" / "test_source3.tsv",
	)
	bundle = load_model(model_path)
	candidate_path = output_directory / "candidate_pairs.tsv"
	matching_path = output_directory / "matching_results.tsv"
	first_write = True
	for source1_chunk in iter_tsv_chunks(dataset_root / "test" / "test_source1.tsv", chunksize=chunk_size):
		candidate_pairs, blocking_stats = blocker.generate(source1_chunk)
		reference = blocker.fetch_records(candidate_pairs["candidate_entity_id"].unique().tolist())
		features = build_pair_features(source1_chunk, reference, candidate_pairs)
		scored_pairs = score_candidates(bundle, features)
		candidate_map = candidate_pairs.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()
		matches = predictions_at_threshold(scored_pairs, threshold)
		candidate_frame = pd.DataFrame({
			"source1_entity_id": source1_chunk["entity_id"].astype(str),
			"candidate_entity_ids": [",".join(sorted(candidate_map.get(str(entity_id), set()))) for entity_id in source1_chunk["entity_id"]],
		})
		matching_frame = pd.DataFrame({
			"source1_entity_id": source1_chunk["entity_id"].astype(str),
			"matched_entity_ids": [",".join(sorted(matches.get(str(entity_id), set()))) for entity_id in source1_chunk["entity_id"]],
		})
		candidate_frame.to_csv(candidate_path, sep="\t", index=False, mode="w" if first_write else "a", header=first_write)
		matching_frame.to_csv(matching_path, sep="\t", index=False, mode="w" if first_write else "a", header=first_write)
		first_write = False
		print({"processed_source1": len(source1_chunk), "candidates": len(candidate_pairs), "blocking": blocking_stats}, flush=True)
	blocker.close()
	return candidate_path, matching_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate entity-resolution predictions from a saved model.")
	parser.add_argument("--dataset-root", default="dataset")
	parser.add_argument("--model-path", default="models/sample_baseline.joblib")
	parser.add_argument("--output-dir", default="output/sample")
	parser.add_argument("--sample-source1", type=int, default=None)
	parser.add_argument("--reference-limit", type=int, default=None)
	parser.add_argument("--threshold", type=float, default=0.3)
	parser.add_argument("--chunked", action="store_true")
	parser.add_argument("--chunk-size", type=int, default=50_000)
	parser.add_argument("--reference-db", default="models/test_reference.sqlite")
	parser.add_argument("--engine", choices=["sqlite", "duckdb"], default="sqlite")
	args = parser.parse_args()
	if args.chunked:
		prediction_runner = run_duckdb_prediction if args.engine == "duckdb" else run_chunked_prediction
		prediction_runner(
		 dataset_root=args.dataset_root,
		 model_path=args.model_path,
		 output_directory=args.output_dir,
		 **({"database_path": args.reference_db} if args.engine == "duckdb" else {"reference_db": args.reference_db}),
		 chunk_size=args.chunk_size,
		 threshold=args.threshold,
		 **({} if args.engine == "duckdb" else {"sample_source1": args.sample_source1, "reference_limit": args.reference_limit}),
	)
	else:
		run_prediction(
		 dataset_root=args.dataset_root,
		 model_path=args.model_path,
		 output_directory=args.output_dir,
		 sample_source1=args.sample_source1,
		 reference_limit=args.reference_limit,
		 threshold=args.threshold,
	)


if __name__ == "__main__":
	main()
