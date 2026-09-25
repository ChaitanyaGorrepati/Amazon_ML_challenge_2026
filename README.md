# Amazon ML Challenge 2026

Baseline implementation for business entity resolution using only the supplied TSV files. Source 1 is the reference entity set; Source 2 and Source 3 contain records that may match zero, one, or multiple Source 1 entities.

## Dataset

The expected layout is:

```text
dataset/
	train/{train_source1.tsv,train_source2.tsv,train_source3.tsv,train_ground_truth.tsv}
	test/{test_source1.tsv,test_source2.tsv,test_source3.tsv}
```

All files are read with `sep="\t"`. The observed source columns are `entity_id`, `business_name`, `business_address`, and `country`. The country field is open-set: the training data contains US and India, while the test data also contains France.

## Pipeline

1. `src/data_loader.py` reads source files and ground truth without changing the dataset.
2. `src/preprocessing.py` retains raw values and adds Unicode-normalized names, addresses, countries, numeric tokens, house numbers, unit numbers, and postal codes.
3. `src/blocking.py` builds country-aware inverted indexes and unions exact, token, and configurable 3/4/5-character n-gram candidates.
4. `src/features.py` computes name, address, structured, numeric, and combined similarity features only for blocked pairs.
5. `src/train.py` labels blocked pairs from ground truth and uses LightGBM when installed, with a scikit-learn histogram gradient boosting fallback.
6. `src/evaluate.py` evaluates macro-averaged entity-level F0.5, including the required zero-match behavior, and sweeps thresholds.
7. `src/predict.py` scores candidates and writes one output row for every Source 1 entity.

## Smoke test before training

The modules are importable and do not start training on import. This small check exercises loading, normalization, blocking, and feature generation without fitting a model:

```powershell
python -c "from src.data_loader import load_sources,load_reference_sources; from src.blocking import CandidateBlocker; from src.features import build_pair_features; train=load_sources('dataset/train','train',nrows=40); reference=load_reference_sources('dataset/train','train',nrows=40); pairs,stats=CandidateBlocker().fit(reference).generate(train['source1'].head(10)); features=build_pair_features(train['source1'].head(10),reference,pairs); print(stats); print(features.shape)"
```

For larger experiments, use `BlockingConfig` to change posting and candidate limits. Always measure blocking recall against `train_ground_truth.tsv` before training.

## Training and validation

Build normalized data, generate candidates, build pair features, and label them with `make_training_frame`. Split by `source1_entity_id` using `split_by_source1`, fit with `fit_model`, and persist with `save_model`. Use `threshold_sweep` on held-out Source 1 entities to select a threshold; do not tune against test data.

The official score is macro F0.5 over Source 1 entities. For an entity with no true matches, predicting no matches scores 1.0 and predicting any match scores 0.0.

## Prediction outputs

`write_submission` writes tab-separated files:

```text
output/candidate_pairs.tsv
output/matching_results.tsv
```

Each has exactly one row per test Source 1 entity. Comma-separated IDs are values inside the TSV field, not field separators. Final matches are selected only from the candidate pairs.

## Dependencies and performance

Install the packages in `requirements.txt`. LightGBM is optional at runtime because the training module falls back to scikit-learn when it is unavailable. The provided data is large, so begin with `nrows` smoke tests, then use chunked loading and measured blocking settings. The implementation does not use external data lookup or augmentation.
