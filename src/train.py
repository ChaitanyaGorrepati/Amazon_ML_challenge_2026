"""Model training utilities; importing this module never starts training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

from .blocking import BlockingConfig, CandidateBlocker, evaluate_blocking_recall
from .data_loader import iter_tsv_chunks, load_ground_truth, read_tsv
from .evaluate import save_experiment, threshold_sweep
from .features import FEATURE_COLUMNS
from .features import build_pair_features


def make_training_frame(features: pd.DataFrame, ground_truth: dict[str, set[str]]) -> pd.DataFrame:
	training = features.copy()
	training["target"] = [
		float(candidate_id in ground_truth.get(source1_id, set()))
		for source1_id, candidate_id in zip(training["source1_entity_id"], training["candidate_entity_id"])
	]
	return training


def split_by_source1(
	training: pd.DataFrame,
	validation_fraction: float = 0.2,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	if training.empty:
		return training.copy(), training.copy()
	splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=random_state)
	train_indices, validation_indices = next(
		splitter.split(training, groups=training["source1_entity_id"])
	)
	return training.iloc[train_indices].copy(), training.iloc[validation_indices].copy()


def _create_classifier(random_state: int = 42) -> tuple[Any, str]:
	try:
		from lightgbm import LGBMClassifier

		return (
			LGBMClassifier(
				objective="binary",
				n_estimators=300,
				learning_rate=0.05,
				num_leaves=31,
				random_state=random_state,
				n_jobs=-1,
			),
			"lightgbm",
		)
	except ImportError:
		return (
			HistGradientBoostingClassifier(
				max_iter=200,
				learning_rate=0.08,
				max_leaf_nodes=31,
				random_state=random_state,
			),
			"sklearn_hist_gradient_boosting",
		)


def fit_model(training: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
	if training.empty:
		raise ValueError("Cannot train a model from an empty candidate set")
	if training["target"].nunique() < 2:
		raise ValueError("Training candidates must contain both positive and negative examples")
	classifier, backend = _create_classifier(random_state)
	classifier.fit(training[FEATURE_COLUMNS], training["target"])
	return {"model": classifier, "feature_columns": FEATURE_COLUMNS, "backend": backend}


def save_model(bundle: dict[str, Any], path: str | Path) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(bundle, path)


def load_model(path: str | Path) -> dict[str, Any]:
	return joblib.load(path)


def predict_probabilities(bundle: dict[str, Any], features: pd.DataFrame) -> pd.Series:
	if features.empty:
		return pd.Series(dtype=float, index=features.index)
	probabilities = bundle["model"].predict_proba(features[bundle["feature_columns"]])[:, 1]
	return pd.Series(probabilities, index=features.index, name="match_probability")


def _sample_reference(
	train_directory: str | Path,
	positive_ids: set[str],
	negative_per_source: int,
) -> pd.DataFrame:
	frames = []
	for source_number in (2, 3):
		path = Path(train_directory) / f"train_source{source_number}.tsv"
		source_positive_ids = {entity_id for entity_id in positive_ids if entity_id.startswith(f"S{source_number}-")}
		selected = []
		negative_count = 0
		for chunk in iter_tsv_chunks(path):
			positive_rows = chunk[chunk["entity_id"].isin(positive_ids)]
			if not positive_rows.empty:
				selected.append(positive_rows.assign(record_source=f"source{source_number}"))
			if negative_count < negative_per_source:
				negative_rows = chunk[~chunk["entity_id"].isin(positive_ids)].head(negative_per_source - negative_count)
				if not negative_rows.empty:
					selected.append(negative_rows.assign(record_source=f"source{source_number}"))
					negative_count += len(negative_rows)
			selected_ids = set(pd.concat(selected)["entity_id"]) if selected else set()
			if negative_count >= negative_per_source and source_positive_ids.issubset(selected_ids):
				break
		frames.extend(selected)
	if not frames:
		raise ValueError("No reference records were selected for the sample")
	return pd.concat(frames, ignore_index=True).drop_duplicates("entity_id")


def run_sample_training(
	dataset_root: str | Path = "dataset",
	sample_source1: int = 1000,
	negative_reference_per_source: int = 5000,
	model_path: str | Path = "models/sample_baseline.joblib",
	random_state: int = 42,
) -> dict[str, Any]:
	dataset_root = Path(dataset_root)
	train_directory = dataset_root / "train"
	source1 = read_tsv(train_directory / "train_source1.tsv", nrows=sample_source1)
	ground_truth = load_ground_truth(train_directory / "train_ground_truth.tsv")
	selected_truth = {str(entity_id): ground_truth.get(str(entity_id), set()) for entity_id in source1["entity_id"]}
	positive_ids = set().union(*selected_truth.values()) if selected_truth else set()
	reference = _sample_reference(train_directory, positive_ids, negative_reference_per_source)
	blocker = CandidateBlocker(
		BlockingConfig(
			name_token_limit=50,
			address_token_limit=50,
			character_ngram_limit=50,
			max_candidates_per_entity=500,
		)
	)
	candidate_pairs, blocking_stats = blocker.fit(reference).generate(source1)
	blocking_stats["candidate_recall"] = evaluate_blocking_recall(candidate_pairs, selected_truth)
	features = build_pair_features(source1, reference, candidate_pairs)
	training = make_training_frame(features, selected_truth)
	train_part, validation_part = split_by_source1(training, random_state=random_state)
	bundle = fit_model(train_part, random_state=random_state)
	validation_scored = validation_part.copy()
	validation_scored["match_probability"] = predict_probabilities(bundle, validation_part)
	thresholds = [round(value, 2) for value in list(np.arange(0.30, 0.96, 0.05))]
	threshold_results = threshold_sweep(
		validation_scored,
		selected_truth,
		validation_part["source1_entity_id"].unique(),
		thresholds,
	)
	best_row = threshold_results.sort_values("macro_f0_5", ascending=False).iloc[0]
	save_model(bundle, model_path)
	experiment = {
		"sample_source1": sample_source1,
		"reference_rows": len(reference),
		"candidate_rows": len(candidate_pairs),
		"feature_rows": len(features),
		"positive_pairs": int(training["target"].sum()),
		"blocking": blocking_stats,
		"model_backend": bundle["backend"],
		"best_threshold": float(best_row["threshold"]),
		"validation": best_row.to_dict(),
	}
	save_experiment(experiment)
	print(experiment)
	return {"bundle": bundle, "threshold_results": threshold_results, "experiment": experiment}


def main() -> None:
	parser = argparse.ArgumentParser(description="Train a small baseline before full-dataset training.")
	parser.add_argument("--dataset-root", default="dataset")
	parser.add_argument("--sample-source1", type=int, default=1000)
	parser.add_argument("--negative-reference-per-source", type=int, default=5000)
	parser.add_argument("--model-path", default="models/sample_baseline.joblib")
	args = parser.parse_args()
	run_sample_training(
		dataset_root=args.dataset_root,
		sample_source1=args.sample_source1,
		negative_reference_per_source=args.negative_reference_per_source,
		model_path=args.model_path,
	)


if __name__ == "__main__":
	main()
