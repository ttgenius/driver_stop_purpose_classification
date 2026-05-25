 # Driver Stop Purpose Classification

This repository contains dataset creation, classification, feature-engineering and performance analysis code for driver stop purpose classification. It was developed as part of Yuezhang Zhu's Master's thesis at the University of Canterbury. We publicly release it support future resarch in transportation engineering and mobility analysis.

The code creates and uses the publicly released dataset Driver Stop Purpose Classification, and conducts various experiments to improve the performance of the binary classification task - whether a driver stop is work or non-work related.

## Acknowledgements

This research was supervised by Professor Richard Green and Dr Nathan Robinson. The research was supported by Chief Scientist Francesco Sambo, Data Scientist Aurel Pjetri, Senior Data Scientist Hughan Ross, and the leadership and legal teams at Verizon Connect.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `code/1_processing_features` | Developing time, poi, road, intersection, and NAICS-based features for the publicly released Driver Stop Purpose Classification dataset |
| `code/2_classification` | Classifying driver stop purpose – whether a driver stop is work or non-work related using different machine learning models with different strategies, including primary evaluation, class imbalance mitigation, error analysis, and feature engineering |
| `code/3_analysis` | Analysing the classification performance and creating summary metrics |
| `input` | Input dataset folder for classification. Contains example CSV shards and metadata. Download the [full dataset published on Kaggle](https://www.kaggle.com/datasets/yuezhangzhu/driver-stop-purpose-classification-dataset/data) to conduct the classification task|
| `output` | Generated models, metrics, plots, tables, and diagnostics. This folder is ignored by git. |
| `requirements.txt` | Python dependency pins used by the local virtual environment. |

## Dataset

The classifiers in `code/2_classification` expect the full Driver Stop Purpose Classification dataset containing 2 million rows under `input`. Download the [full dataset](https://www.kaggle.com/datasets/yuezhangzhu/driver-stop-purpose-classification-dataset/data) and extract into the following structure:

```text
input/
  stop_records/
    anonymised_part_001.csv
    ...
  stop_records_with_time_features/
    anonymised_part_001_with_time_features.csv
    ...
  stop_records_with_poi_features/
    anonymised_part_001_with_poi_features.csv
    ...
  stop_records_with_road_features/
    anonymised_part_001_with_road_features.csv
    ...
  stop_records_with_intersection_features/
    anonymised_part_001_with_intersection_features.csv
    ...
  stop_records_with_enhanced_features/
    anonymised_part_001_with_enhanced_features.csv
    ...
  feature_classification_metadata/
    poi_category_classification.csv
    road_type_classification.csv
```

The full dataset used in the thesis contains 174 matching CSV shards in each
`stop_records*` folder. Matching shard numbers represent the same records in the
same row order, with each feature folder retaining the original columns and
adding one feature family.

`is_work_stop` is the classification label column:

- `1`: work stop
- `0`: non-work stop

Important raw fields:

- `unit_id`: anonymised vehicle identifier used for grouped train/validation/test splitting.
- `POIs`: semicolon-separated OSM POI type and distance pairs.
- `Stop Road Info`: serialized road dictionaries from OSM.
- `Nearest Intersections`: serialized intersection dictionaries from OSM.
- `naics1`, `naics2`, `naics3`, `naics1_long`, `naics2_long`, `naics3_long`: 3-digit and 6-digit code classified from the North American Industry Classification System (NAICS) for the type of business associated with the vehicle.

The modelling code excludes raw identifiers, timestamps, raw serialized OSM
fields, planned arrival/departure times, `distance_to_work_stop`, and raw NAICS codes from model features.

Two dataset configurations are defined in the thesis experiments:

1. **Baseline dataset**
The baseline dataset consists of 75 features, including time, POI, road, and intersection features. The baseline dataset is merged by folder `stop_records_with_time_features`, `stop_records_with_poi_features`, `stop_records_with_road_features` and `stop_records_with_intersection_features`.

2. **Enhanced dataset**
The enhanced dataset consisting of 112 features, extending the baseline dataset by adding the enhanced context features developed in Chapter 6, and is used for the final proposed machine learning pipeline. To use the enhanced dataset, you can run `enhanced_classifier.py` to generate all the enhanced features during experimentation, without merging the **baseline dataset** with `stop_records_with_enhanced_features`. Note the two NAICS frequency prior features used in the final machine learning pipeline are generated dynamically during training to avoid data leakage, therefore not included in the `stop_records_with_enhanced_features`.

Both dataset configurations contain the same vehicle stops, labels, and anonymisation. They differ only in the available features.

## Environment Setup

The local environment used Python 3.8. Install dependencies from the repository
root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow is configured in the classifier scripts to use GPU if available and
otherwise fall back to CPU.

## Feature Processing

The folder `1_procssing_features` contains scripts to extract raw featues from OpenStreetMap, then further process time, POI, road, intersection features for the Driver Stop Purpose Classification Dataset. These scripts do not need to be run, only for reference purposes. The fully processed Dataset is published and available on [Kaggle](https://www.kaggle.com/datasets/yuezhangzhu/driver-stop-purpose-classification-dataset/data).


### Processing Scripts

| Script | Description |
| --- | --- |
| `get_osm_raw_features/get_osm_raw_features.py` | Helper functions for querying Overpass/OSM around latitude-longitude stops. Adds raw `POIs`, `Stop Road Info`, and `Nearest Intersections` fields. This file is a helper module; call `process_csv(input_csv, output_csv)` from another script or an interactive session. |
| `process_time/add_time_features.py` | Reads `input/stop_records/*.csv` and writes `input/stop_records_with_time_features/*_with_time_features.csv`. Adds seconds-since-midnight, sine/cosine cyclical encodings for stop start/end time, and `is_weekend`. |
| `process_poi/extract_poi_types.py` | Scans raw `POIs` strings and writes distinct POI types to `input/poi_category/stop_records_output/POI_types.csv`. Useful when updating the POI category mapping. |
| `process_poi/extract_all_poi_categories.py` | Extracts all unique category labels from `input/feature_classification_metadata/poi_category_classification.csv`. Supports `--input` and `--output`; default output is `input/poi_category/all_poi_categories.csv`. |
| `process_poi/process_poi_counts_features.py` | Uses `poi_category_classification.csv` to map raw OSM POI types to category counts. Writes `input/stop_records_with_poi_features/*_with_poi_features.csv`, including one column per POI category and `total_pois`. |
| `process_poi/check_missing_pois_percentage.py` | Reports the percentage of rows with missing or blank `POIs` values in `input/stop_records_with_poi_features`. |
| `process_road/analyze_road_features.py` | Exploratory script for inspecting `Stop Road Info` structure, road types, speeds, access values, lanes, and distances from sample files. |
| `process_road/extract_road_types.py` | Scans all raw stop records for OSM `road_type` values and writes summary files under `input/road_types`, including counts, per-file statistics, and a JSON summary. |
| `process_road/process_road_features.py` | Uses `road_type_classification.csv` to derive road-category counts, speed-limit features, lane features, road availability flags, and distance aggregates. Writes `input/stop_records_with_road_features/*_with_road_features.csv`. |
| `process_intersection/extract_intersection_types.py` | Extracts distinct `intersection_type` values from `Nearest Intersections` and writes `input/intesection_types/intersection_types.csv` (folder spelling matches the script). |
| `process_intersection/process_intersection_features.py` | Parses `Nearest Intersections` in parallel and writes `input/stop_records_with_intersection_features/*_with_intersection_features.csv`. Adds intersection counts, distances, sign counts, speed features, lane flags, crossing/railway/barrier indicators. |

## Classification

There are two classifiers used in different stages of the thesis experiemnts

| Script | Purpose |
| --- | --- |
| `code/2_classification/baseline_classifier.py` | Chapter 4 primary evaluation. Compares model performance across baseline feature combinations using group-aware train/validation/test splits by `unit_id` with the baseline dataset. Saves trained models and metrics. |
| `code/2_classification/enhanced_classifier.py` | Chapter 5 and Chapter 6 experiments. Evaluates class-imbalance strategies for LightGBM and Neural Network, conducts false-positive analysis, and creates enhanced POI/duration/NAICS features for the enhanced dataset. Develops the final proposed machine learning pipeline. |

Available feature modes:

| Mode | Feature family |
| --- | --- |
| `time_only` | Time and duration features only. |
| `poi_only` | POI category counts only. |
| `road_only` | Road features only. |
| `intersection_only` | Intersection features only. |
| `poi_time` | POI plus time features. |
| `road_time` | Road plus time features. |
| `intersection_time` | Intersection plus time features. |
| `all` | Time, POI, road, and intersection features. In the enhanced classifier, this also includes engineered POI fraction, duration-context, and NAICS features. |

Available model selectors:

| Selector | Model |
| --- | --- |
| `lr` | Logistic Regression |
| `rf` | Random Forest |
| `lgbm` | LightGBM |
| `nn` | Dense neural network |
| `cnn` | 1D convolutional neural network over tabular features |
| `tbt` | Transformer-inspired tabular neural network |

### Run the Baseline Classifier

From the repository root:

```powershell
python code\2_classification\baseline_classifier.py
```

The current `main()` defaults to:

```python
feature_modes = ['all']
results = classifier.run_baseline_pipeline(models=['lgbm'])
```

To compare feature modes, edit `feature_modes`, for example:

```python
feature_modes = ['time_only', 'poi_time', 'road_time', 'intersection_time', 'all']
```

To compare models, edit the `models` list:

```python
results = classifier.run_baseline_pipeline(models=['lr', 'rf', 'lgbm'])
```

If `models=None`, `run_baseline_pipeline()` attempts all supported baseline
models: `rf`, `lr`, `nn`, `cnn`, `lgbm`, and `tbt`.

### Run the Enhanced Classifier

From the repository root:

```powershell
python code\2_classification\enhanced_classifier.py
```

The current `main()` defaults to the final proposed machine learning pipeline - LightGBM with threshold tuning at minimum recall 0.85 to maximize specificity, using the enhanced dataset.

```python
feature_modes = ['all']
results = classifier.run_enhanced_pipeline(min_recalls=[0.85], models=['lgbm'])
```

Key experiment constants are defined near the top of `enhanced_classifier.py`:

| Constant | Default | Purpose |
| --- | ---: | --- |
| `MIN_RECALL` | `0.85` | Default minimum recall constraint used by threshold tuning. |
| `RANDOM_STATE` | `42` | Reproducibility seed used for grouped splitting, training, downsampling, and error analysis. Change to other seeds to assess performance stability |
| `SAVE_ENHANCED_FILES` | `False` | Controls whether enhanced feature CSV files are exported by default when loading/merging the enhanced data. |

For the LightGBM experiments, the script evaluates three class imbalance mitigation strategies:

- threshold tuning: maximises specificity subject to a minimum recall constraint;
- class weighting: optionally sets LightGBM `scale_pos_weight` from the training split as `n_negative / n_positive`;
- probability calibration: optionally calibrates LightGBM probabilities on the validation split before threshold tuning, using `sigmoid` Platt scaling or `isotonic` regression.

The current final-style `run_enhanced_pipeline()` call uses threshold tuning and
sets `use_scale_pos_weight=False` and `calibrate_probabilities=False`. To test
multiple recall constraints:

```python
results = classifier.run_enhanced_pipeline(
    min_recalls=[0.95, 0.90, 0.85, 0.80],
    models=['lgbm']
)
```

To enable LightGBM class weighting or probability calibration, edit the
`self.train_lightgbm(...)` call inside `run_enhanced_pipeline()`:

```python
self.train_lightgbm(
    X_train, X_val, X_test, y_train, y_val, y_test,
    use_scale_pos_weight=True,
    tune_threshold=True,
    min_recall=min_recall,
    calibrate_probabilities=True,
    calibration_method="sigmoid",  # or "isotonic"
    train_downsample=False
)
```

For neural-network imbalance experiments:

```python
results = classifier.run_enhanced_pipeline(
    min_recalls=[0.85],
    models=['nn'],
    nn_imbalance_modes=[
        'baseline',
        'class_weight',
        'focal',
        'balanced_batches',
        'balanced_focal',
    ]
)
```

Enhanced feature export is controlled by `SAVE_ENHANCED_FILES`, with the current `SAVE_ENHANCED_FILES = False`, the enhanced features are created in memory. To export those enhanced features to CSVs, set the module-level constant:

```python
SAVE_ENHANCED_FILES = True
```

### False-Positive Analysis

`enhanced_classifier.py` can reload a saved model and threshold for false
positive versus true negative diagnostics. In the `main()` block, comment out
the training call and use the helper sequence:

```python
classifier.prepare_splits_only(seed=RANDOM_STATE)
classifier.load_trained_model(r"output\ml_results\saved_ml_models\MODEL_FILE.pkl")
classifier.load_decision_threshold_from_metrics(
    r"output\ml_results\saved_ml_performance_metrics\METRICS_FILE.json"
)
classifier.run_fp_diagnosis(
    seed=RANDOM_STATE,
    out_dir=r"output\fp_diagnostics",
    split="test"
)
```

## Outputs

Classifier outputs are written under `output/ml_results`:

```text
output/ml_results/
  saved_ml_models/
    *.pkl
    *.h5
  saved_ml_performance_metrics/
    *_metrics.json
  tables/
    *_threshold_sweep_*.csv
  plots/
    *.png
  fp_diagnostics/
```

The metrics JSON includes training metadata, selected feature mode, dataset
size, number of features, test metrics, feature importance when available, and
model hyperparameters. Under threshold tuning, the selected decision
threshold is stored in the metrics hyperparameters.

## Evaluation Notes

- Splits are group-aware by `unit_id`, reducing leakage from the same vehicle
  appearing in train and test data.
- Avoid using planned arrival/departure times, `distance_to_work_stop`, and raw 3-digit and 6-digit NAICS columns in classification unless for specific requirements.
- Report recall, specificity, precision, F1, balanced accuracy, ROC-AUC, and PR-AUC instead of accuracy for this imbalanced classification task.


## License

This project is licensed under the MIT License. See the LICENSE file for details.