"""Entity-level F0.5 and blocking evaluation helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
	if precision == 0.0 and recall == 0.0:
		return 0.0
	beta_squared = beta * beta
	return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def evaluate_predictions(
	predictions: dict[str, set[str]],
	ground_truth: dict[str, set[str]],
	source1_ids: Iterable[str] | None = None,
	beta: float = 0.5,
) -> dict[str, float]:
	entity_ids = list(source1_ids if source1_ids is not None else ground_truth.keys())
	scores = []
	precisions = []
	recalls = []
	singleton_correct = 0
	predicted_count = 0
	for entity_id in entity_ids:
		predicted = set(predictions.get(entity_id, set()))
		truth = set(ground_truth.get(entity_id, set()))
		predicted_count += len(predicted)
		if not truth:
			singleton_correct += int(not predicted)
			scores.append(1.0 if not predicted else 0.0)
			precisions.append(1.0 if not predicted else 0.0)
			recalls.append(1.0 if not predicted else 0.0)
			continue
		true_positives = len(predicted & truth)
		precision = true_positives / len(predicted) if predicted else 0.0
		recall = true_positives / len(truth)
		precisions.append(precision)
		recalls.append(recall)
		scores.append(fbeta(precision, recall, beta=beta))
	count = len(entity_ids)
	return {
		"macro_f0_5": sum(scores) / count if count else 0.0,
		"macro_precision": sum(precisions) / count if count else 0.0,
		"macro_recall": sum(recalls) / count if count else 0.0,
		"singleton_accuracy": singleton_correct / count if count else 0.0,
		"predicted_matches": float(predicted_count),
		"entities": float(count),
	}


def predictions_at_threshold(
	scored_pairs: pd.DataFrame,
	threshold: float,
) -> dict[str, set[str]]:
	selected = scored_pairs[scored_pairs["match_probability"] >= threshold]
	return selected.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()


def threshold_sweep(
	scored_pairs: pd.DataFrame,
	ground_truth: dict[str, set[str]],
	source1_ids: Iterable[str],
	thresholds: Iterable[float],
) -> pd.DataFrame:
	rows = []
	for threshold in thresholds:
		metrics = evaluate_predictions(
			predictions_at_threshold(scored_pairs, threshold),
			ground_truth,
			source1_ids=source1_ids,
		)
		rows.append({"threshold": threshold, **metrics})
	return pd.DataFrame(rows)


def candidate_recall(candidate_pairs: pd.DataFrame, ground_truth: dict[str, set[str]]) -> float:
	grouped = candidate_pairs.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()
	covered = [truth.issubset(grouped.get(entity_id, set())) for entity_id, truth in ground_truth.items()]
	return sum(covered) / len(covered) if covered else 1.0


def save_experiment(metrics: dict[str, object], directory: str | Path = "experiments") -> Path:
	directory = Path(directory)
	directory.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
	path = directory / f"experiment_{timestamp}.json"
	path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
	return path
