"""Pairwise feature engineering for candidate entity pairs."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import math

import pandas as pd

try:
	from rapidfuzz import fuzz
except ImportError:
	fuzz = None

from .preprocessing import add_normalized_columns


FEATURE_COLUMNS = [
	"name_token_jaccard",
	"name_token_sort_ratio",
	"name_partial_ratio",
	"name_levenshtein_similarity",
	"name_character_similarity",
	"name_tfidf_cosine",
	"address_token_jaccard",
	"address_partial_ratio",
	"address_levenshtein_similarity",
	"address_character_similarity",
	"address_tfidf_cosine",
	"country_match",
	"exact_number_match",
	"postal_code_match",
	"numeric_token_overlap",
	"name_length_difference",
	"address_length_difference",
	"combined_tfidf_cosine",
]


def _tokens(value: str) -> set[str]:
	return set(value.split()) if value else set()


def _token_jaccard(left: str, right: str) -> float:
	left_tokens, right_tokens = _tokens(left), _tokens(right)
	if not left_tokens and not right_tokens:
		return 1.0
	return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens | right_tokens else 0.0


def _sequence_ratio(left: str, right: str) -> float:
	if fuzz is not None:
		return fuzz.ratio(left, right) / 100.0 if left or right else 1.0
	return SequenceMatcher(None, left, right).ratio() if left or right else 1.0


def _partial_ratio(left: str, right: str) -> float:
	if not left or not right:
		return 1.0 if left == right else 0.0
	if fuzz is not None:
		return fuzz.partial_ratio(left, right) / 100.0
	shorter, longer = sorted((left, right), key=len)
	if shorter in longer:
		return 1.0
	window = len(shorter)
	return max(SequenceMatcher(None, shorter, longer[start : start + window]).ratio() for start in range(len(longer) - window + 1))


def _character_similarity(left: str, right: str) -> float:
	def grams(value: str) -> set[str]:
		padded = f"  {value}  "
		return {padded[index : index + 3] for index in range(max(0, len(padded) - 2))}

	left_grams, right_grams = grams(left), grams(right)
	if not left_grams and not right_grams:
		return 1.0
	return len(left_grams & right_grams) / len(left_grams | right_grams) if left_grams | right_grams else 0.0


def _cosine(left: str, right: str) -> float:
	left_counts, right_counts = Counter(left.split()), Counter(right.split())
	if not left_counts and not right_counts:
		return 1.0
	common = set(left_counts) & set(right_counts)
	numerator = sum(left_counts[token] * right_counts[token] for token in common)
	denominator = math.sqrt(sum(value * value for value in left_counts.values())) * math.sqrt(sum(value * value for value in right_counts.values()))
	return numerator / denominator if denominator else 0.0


def _numeric_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
	left_set, right_set = set(left), set(right)
	if not left_set and not right_set:
		return 1.0
	return len(left_set & right_set) / len(left_set | right_set) if left_set | right_set else 0.0


def _pair_features(left: object, right: object) -> dict[str, float | str]:
	name_left, name_right = left.name_normalized, right.name_normalized
	address_left, address_right = left.address_normalized, right.address_normalized
	combined_left = f"{name_left} {address_left}".strip()
	combined_right = f"{name_right} {address_right}".strip()
	return {
		"source1_entity_id": str(left.entity_id),
		"candidate_entity_id": str(right.entity_id),
		"name_token_jaccard": _token_jaccard(name_left, name_right),
		"name_token_sort_ratio": _sequence_ratio(" ".join(sorted(_tokens(name_left))), " ".join(sorted(_tokens(name_right)))),
		"name_partial_ratio": _partial_ratio(name_left, name_right),
		"name_levenshtein_similarity": _sequence_ratio(name_left, name_right),
		"name_character_similarity": _character_similarity(name_left, name_right),
		"name_tfidf_cosine": _cosine(name_left, name_right),
		"address_token_jaccard": _token_jaccard(address_left, address_right),
		"address_partial_ratio": _partial_ratio(address_left, address_right),
		"address_levenshtein_similarity": _sequence_ratio(address_left, address_right),
		"address_character_similarity": _character_similarity(address_left, address_right),
		"address_tfidf_cosine": _cosine(address_left, address_right),
		"country_match": float(left.country_normalized == right.country_normalized),
		"exact_number_match": float(bool(left.house_number) and left.house_number == right.house_number),
		"postal_code_match": float(bool(left.postal_code) and left.postal_code == right.postal_code),
		"numeric_token_overlap": _numeric_overlap(left.address_numeric_tokens, right.address_numeric_tokens),
		"name_length_difference": abs(len(name_left) - len(name_right)),
		"address_length_difference": abs(len(address_left) - len(address_right)),
		"combined_tfidf_cosine": _cosine(combined_left, combined_right),
	}


def build_pair_features(
	source1: pd.DataFrame,
	reference: pd.DataFrame,
	candidate_pairs: pd.DataFrame,
) -> pd.DataFrame:
	if candidate_pairs.empty:
		return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", *FEATURE_COLUMNS])
	left = add_normalized_columns(source1).set_index("entity_id", drop=False)
	right = add_normalized_columns(reference).set_index("entity_id", drop=False)
	rows = []
	for pair in candidate_pairs.itertuples(index=False):
		if pair.source1_entity_id not in left.index or pair.candidate_entity_id not in right.index:
			continue
		rows.append(_pair_features(left.loc[pair.source1_entity_id], right.loc[pair.candidate_entity_id]))
	return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", *FEATURE_COLUMNS])
