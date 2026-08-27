# Graph Report - .  (2026-08-03)

## Corpus Check
- Corpus is ~8,740 words - fits in a single context window. You may not need a graph.

## Summary
- 115 nodes · 188 edges · 9 communities (8 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Anomaly Detection Models|Anomaly Detection Models]]
- [[_COMMUNITY_Feature Engineering|Feature Engineering]]
- [[_COMMUNITY_Data Loading|Data Loading]]
- [[_COMMUNITY_Evaluation Metrics|Evaluation Metrics]]
- [[_COMMUNITY_Model Serving API|Model Serving API]]
- [[_COMMUNITY_Pipeline Runner|Pipeline Runner]]
- [[_COMMUNITY_API Smoke Tests|API Smoke Tests]]
- [[_COMMUNITY_Synthetic Data Generation|Synthetic Data Generation]]
- [[_COMMUNITY_Notebook Builder|Notebook Builder]]

## God Nodes (most connected - your core abstractions)
1. `DataFrame` - 9 edges
2. `_engineer_common()` - 9 edges
3. `evaluate_subdataset()` - 8 edges
4. `SubDataset` - 7 edges
5. `load_subdataset()` - 7 edges
6. `engineer_features()` - 6 edges
7. `AutoencoderDetector` - 6 edges
8. `process_one()` - 6 edges
9. `_read_csv_robust()` - 5 edges
10. `load_all()` - 5 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (9 total, 1 thin omitted)

### Community 0 - "Anomaly Detection Models"
Cohesion: 0.13
Nodes (14): Pipeline, AutoencoderDetector, calibrate_threshold(), IsolationForestDetector, make_preprocessor(), DataFrame, ndarray, model.py ========= Two anomaly-detection approaches, trained per-turbine on that (+6 more)

### Community 1 - "Feature Engineering"
Cohesion: 0.16
Nodes (21): add_rolling_features(), apply_power_curve_reference(), build_label(), clean_zeros_as_missing(), _engineer_common(), engineer_features(), engineer_features_for_serving(), fit_power_curve_reference() (+13 more)

### Community 2 - "Data Loading"
Cohesion: 0.19
Nodes (14): discover_subdatasets(), _find_event_metadata(), load_all(), load_subdataset(), DataFrame, Path, data_loader.py =============== Loading, validating, and lightly cleaning individ, Find candidate SCADA CSVs under raw_dir. Skips files that are clearly     metada (+6 more)

### Community 3 - "Evaluation Metrics"
Cohesion: 0.23
Nodes (12): aggregate_results(), _count_events(), evaluate_subdataset(), _fbeta(), DataFrame, ndarray, Series, evaluation.py ============== Evaluation aligned with the CARE benchmark's own sc (+4 more)

### Community 4 - "Model Serving API"
Cohesion: 0.30
Nodes (11): BaseModel, _check_auth(), health(), _list_available_models(), list_models(), _load_bundle(), predict(), PredictRequest (+3 more)

### Community 5 - "Pipeline Runner"
Cohesion: 0.31
Nodes (8): _fmt(), main(), process_one(), Path, run_pipeline.py ================ End-to-end script: for each Wind Farm A sub-dat, Run the full fit/score/evaluate cycle for a single sub-dataset., SubDataset, SubDatasetResult

### Community 6 - "API Smoke Tests"
Cohesion: 0.38
Nodes (5): config.py ========= Central configuration for the Wind Farm A early-fault-detect, _get(), main(), _post(), smoke_test_api.py ================== End-to-end smoke test for the serving API:

### Community 7 - "Synthetic Data Generation"
Cohesion: 0.57
Nodes (6): build_synthetic_farm(), make_anomaly_dataset(), make_normal_dataset(), _make_series(), Path, make_synthetic_data.py ======================== Generates small synthetic sub-da

## Knowledge Gaps
- **5 isolated node(s):** `Series`, `Timestamp`, `Pipeline`, `SubDataset`, `SubDatasetResult`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `config.py ========= Central configuration for the Wind Farm A early-fault-detect`, `data_loader.py =============== Loading, validating, and lightly cleaning individ`, `Container for one turbine's train+prediction episode.` to the rest of the system?**
  _37 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Anomaly Detection Models` be split into smaller, more focused modules?**
  _Cohesion score 0.12648221343873517 - nodes in this community are weakly interconnected._