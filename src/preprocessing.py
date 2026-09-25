"""Reusable normalization and structured-text extraction helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_NUMBER_RE = re.compile(r"\b\d+[a-z]?\b", flags=re.IGNORECASE)
_POSTAL_RE = re.compile(r"\b\d{4,10}\b")
_UNIT_RE = re.compile(r"\b(?:apt|apartment|unit|suite|ste|floor|fl)\s*([\w-]+)")

_LEGAL_SUFFIXES = {
	"private limited": "pvt ltd",
	"private ltd": "pvt ltd",
	"pvt limited": "pvt ltd",
	"pvt ltd": "pvt ltd",
	"p ltd": "pvt ltd",
	"public limited": "public ltd",
	"public ltd": "public ltd",
	"limited": "ltd",
	"corporation": "corp",
	"incorporated": "inc",
	"company": "co",
}

_ADDRESS_REPLACEMENTS = {
	"street": "st",
	"str": "st",
	"road": "rd",
	"avenue": "ave",
	"boulevard": "blvd",
	"drive": "dr",
	"lane": "ln",
	"place": "pl",
	"highway": "hwy",
	"apartment": "unit",
	"apt": "unit",
	"suite": "unit",
	"ste": "unit",
}


def _clean_unicode(value: Any) -> str:
	if value is None or pd.isna(value):
		return ""
	text = unicodedata.normalize("NFKC", str(value))
	text = text.replace("\u200b", " ").replace("\ufeff", " ")
	text = "".join(char for char in text if unicodedata.category(char) != "Cf")
	text = unicodedata.normalize("NFKD", text)
	return "".join(char for char in text if not unicodedata.combining(char))


def normalize_text(value: Any, remove_punctuation: bool = True) -> str:
	text = _clean_unicode(value).lower()
	if remove_punctuation:
		text = _PUNCTUATION_RE.sub(" ", text)
	return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_name(value: Any) -> str:
	text = normalize_text(value)
	for source, target in sorted(_LEGAL_SUFFIXES.items(), key=lambda item: -len(item[0])):
		text = re.sub(rf"\b{re.escape(source)}\b", target, text)
	return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_address(value: Any) -> str:
	text = normalize_text(value)
	tokens = [_ADDRESS_REPLACEMENTS.get(token, token) for token in text.split()]
	return " ".join(tokens)


def normalize_country(value: Any) -> str:
	return normalize_text(value)


def extract_numeric_tokens(value: Any) -> tuple[str, ...]:
	text = normalize_text(value, remove_punctuation=False)
	return tuple(_NUMBER_RE.findall(text))


def extract_postal_code(value: Any) -> str:
	matches = _POSTAL_RE.findall(normalize_text(value, remove_punctuation=False))
	return matches[-1] if matches else ""


def extract_house_number(value: Any) -> str:
	numbers = extract_numeric_tokens(value)
	return numbers[0] if numbers else ""


def extract_unit_number(value: Any) -> str:
	match = _UNIT_RE.search(normalize_text(value, remove_punctuation=False))
	return match.group(1) if match else ""


def add_normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
	result = frame.copy()
	result["name_normalized"] = result["business_name"].map(normalize_name)
	result["address_normalized"] = result["business_address"].map(normalize_address)
	result["country_normalized"] = result["country"].map(normalize_country)
	result["name_numeric_tokens"] = result["name_normalized"].map(extract_numeric_tokens)
	result["address_numeric_tokens"] = result["address_normalized"].map(extract_numeric_tokens)
	result["postal_code"] = result["address_normalized"].map(extract_postal_code)
	result["house_number"] = result["address_normalized"].map(extract_house_number)
	result["unit_number"] = result["address_normalized"].map(extract_unit_number)
	return result
