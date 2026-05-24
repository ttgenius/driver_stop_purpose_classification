#!/usr/bin/env python3
"""
Enhanced Work Stop Classification for Chapter 5 & 6

This script is for evaluating the Chapter 5 & 6 in Yuezhang Zhu's Master's thesis:
1. All the Class imbalance mitigation methods in Chapter 5
2. False positive error analysis in Chapter 6
3. Feature engineering to create the enhanced features in Chapter 6

This script establish the final proposed machine learning pipeline, using:
1. The enhanced dataset [baselinE features (time, poi, road, intersection) + enhanced features (poi_fraction, duration x poi_fraction, coarse NAICS))
2. LightGBM classification model
3. Threshold tuning that maximises specificity under a minimum recall constraint of 0.85
"""

import os
# Configure for AMD GPU (ROCm) support
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, Input, optimizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, MultiHeadAttention, LayerNormalization, Add,Conv1D, MaxPooling1D,Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
import pickle
import json
from datetime import datetime
import time
import lightgbm as lgb
import numpy as np
import re as _re

MIN_RECALL = 0.85
RANDOM_STATE = 42
REPO_ROOT_DIR = Path(__file__).resolve().parents[2]
BASE_INPUT_DIR = REPO_ROOT_DIR / "input"
RAW_STOP_RECORDS_DIR = Path(BASE_INPUT_DIR) / "stop_records"
ENHANCED_FEATURES_OUTPUT_DIR = Path(BASE_INPUT_DIR) / "stop_records_with_enhanced_features"
BASE_OUTPUT_DIR = REPO_ROOT_DIR / "output"
NUM_FILES_EACH_FOLDER = 174
SAVE_ENHANCED_FILES = False

# Configure TensorFlow for optimal performance
def configure_tensorflow():
    """Configure TensorFlow for optimal performance on available hardware"""
    try:
        # Check for available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            print(tf.sysconfig.get_build_info())

            # # 🔹 Step 2: FORCE TensorFlow to see ONLY GPU(s)
            # # (This is the "force GPU" step)
            # tf.config.set_visible_devices(gpus, 'GPU')

            # Configure memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ GPU memory growth configured for {gpu.name}")
                except Exception as e:
                    print(f"⚠️  Could not configure GPU {gpu.name}: {e}")

            return True
        else:
            print("💻 No TensorFlow-compatible GPU devices detected")
            print("   Hardware: AMD Ryzen 7 PRO 8840HS w/ Radeon 780M Graphics")
            print("   Environment: WSL2 (limited GPU support)")
            print("   Note: AMD GPU requires tensorflow-rocm (not available in WSL2)")
            print("   Optimizing for high-performance CPU computation...")

            # Optimize CPU performance for AMD Ryzen
            tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
            tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores

            # Additional CPU optimizations
            import os
            os.environ['TF_NUM_INTEROP_THREADS'] = '0'
            os.environ['TF_NUM_INTRAOP_THREADS'] = '0'

            print("✅ High-performance CPU optimization configured for AMD Ryzen")
            return False
    except Exception as e:
        print(f"⚠️  TensorFlow configuration failed: {e}")
        print("   Using default settings")
        return False

# Configure TensorFlow
gpu_available = configure_tensorflow()

def display_system_info():
    """Display system and hardware information"""
    import platform
    import psutil

    print("🖥️  SYSTEM INFORMATION")
    print("=" * 50)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    # TensorFlow info
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU Support: {tf.test.is_built_with_gpu_support()}")

    # Check for GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Devices: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("GPU Devices: None detected")

    print("=" * 50)

def add_poi_fraction_features(
    df: pd.DataFrame,
    poi_category_cols,
    total_pois_col: str = "total_pois",
    prefix: str = "poi_frac_",
    add_commercial_frac: bool = True,
):
    """
    Adds POI fraction (semantic density) features:
        poi_frac_<category> = poi_count(category) / max(total_pois, 1)

    Returns:
        (df_with_new_features, new_feature_names)

    Notes:
    - Uses max(total_pois, 1) to avoid division by zero
    - Leaves original POI count columns untouched
    - Works with columns that contain spaces/commas; LightGBM name sanitization can happen later
    """
    out = df.copy()

    if total_pois_col not in out.columns:
        raise KeyError(f"Expected column '{total_pois_col}' not found in df.")

    # Ensure numeric and safe denominator
    total = pd.to_numeric(out[total_pois_col], errors="coerce").fillna(0.0)
    denom = np.maximum(total.values.astype(float), 1.0)

    new_cols: list[str] = []

    # Add category fractions
    for col in poi_category_cols:
        if col not in out.columns:
            # skip silently or warn; I recommend warning during development
            print(f"Warning: POI category column missing, skipping: {col}")
            continue

        counts = pd.to_numeric(out[col], errors="coerce").fillna(0.0).values.astype(float)
        frac = counts / denom
        # clip for safety (can slightly exceed 1 if data issues)
        frac = np.clip(frac, 0.0, 1.0)

        new_name = f"{prefix}{col}"
        out[new_name] = frac
        new_cols.append(new_name)

    # Optional: one aggregated “commercial intensity” fraction
    if add_commercial_frac:
        commercial_components = [
            "Restaurants, Fast Food and Cafes",
            "Shopping and Retail",
            "Gas Stations",
            "Personal Care and Beauty",
            "Entertainment and Recreation",
        ]
        present = [c for c in commercial_components if c in out.columns]
        if len(present) >= 1:
            commercial_sum = np.zeros(len(out), dtype=float)
            for c in present:
                commercial_sum += pd.to_numeric(out[c], errors="coerce").fillna(0.0).values.astype(float)

            commercial_frac = commercial_sum / denom
            commercial_frac = np.clip(commercial_frac, 0.0, 1.0)

            commercial_name = f"{prefix}commercial_frac"
            out[commercial_name] = commercial_frac
            new_cols.append(commercial_name)
        else:
            print("Warning: no commercial POI component columns found; skipping commercial_frac")

    return out, new_cols


def _sanitize_colname(name: str) -> str:
    # must match your LightGBM sanitization later, but here we just create valid-ish feature names
    import re
    return re.sub(r'[^0-9a-zA-Z_]+', '_', name)

def add_duration_context_interactions(
    df: pd.DataFrame,
    duration_col: str = "stop_duration_s",
    poi_frac_cols = None,
    include_duration_bins: bool = True,
    prefix: str = "int_",
):
    """
    Adds duration-context interaction features targeting FP regimes.
    Requires poi_frac_* columns already present in df.

    Returns: (df_out, new_cols)
    """
    out = df.copy()
    new_cols: list[str] = []

    if duration_col not in out.columns:
        raise KeyError(f"Missing required duration column: {duration_col}")

    dur = pd.to_numeric(out[duration_col], errors="coerce").fillna(0.0).astype(float)
    logdur = np.log1p(np.clip(dur.values, 0.0, None))
    out["log_stop_duration_s"] = logdur
    new_cols.append("log_stop_duration_s")

    # Choose default POI fraction columns if not provided
    if poi_frac_cols is None:
        # these are the raw (unsanitized) names you created in add_poi_fraction_features
        candidates = [
            "poi_frac_commercial_frac",
            "poi_frac_Restaurants, Fast Food and Cafes",
            "poi_frac_Shopping and Retail",
            "poi_frac_Transportation and Transit",
            "poi_frac_Parking",
            "poi_frac_Gas Stations",
        ]
        poi_frac_cols = [c for c in candidates if c in out.columns]

    # log(duration) × poi_frac interactions
    for c in poi_frac_cols:
        if c not in out.columns:
            continue
        frac = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float).values
        name = f"{prefix}logdur_x_{_sanitize_colname(c.replace('poi_frac_', ''))}"
        out[name] = logdur * frac
        new_cols.append(name)

    # Optional: duration bins × commercial_frac
    if include_duration_bins and "poi_frac_commercial_frac" in out.columns:
        frac_comm = pd.to_numeric(out["poi_frac_commercial_frac"], errors="coerce").fillna(0.0).astype(float).values

        # bins aligned to your FP diagnostics
        bins = [
            ("durbin_120_600", (dur.values >= 120) & (dur.values < 600)),
            ("durbin_600_1800", (dur.values >= 600) & (dur.values < 1800)),
            ("durbin_1800p", (dur.values >= 1800)),
        ]
        for bname, mask in bins:
            name = f"{prefix}{bname}_x_commercial"
            out[name] = (mask.astype(float) * frac_comm)
            new_cols.append(name)

    return out, new_cols


class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_every=5):
        self.log_every = log_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every == 0:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            print(
                f"Epoch {epoch+1:03d} | "
                f"loss={logs['loss']:.4f} | "
                f"val_loss={logs.get('val_loss', 0):.4f} | "
                f"lr={lr:.6f}"
            )

from typing import Dict, List, Optional

RAW_NAICS_COLS_3D = ["naics1", "naics2", "naics3"]
RAW_NAICS_COLS_6D = ["naics1_long", "naics2_long", "naics3_long"]
RAW_NAICS_COLS = RAW_NAICS_COLS_3D + RAW_NAICS_COLS_6D

def _to_int_safe(x) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(x)
    except Exception:
        return 0

def _naics_to_2d(code: int) -> int:
    """
    Convert NAICS code (3- or 6-digit) to strict 2-digit sector code.
    Returns 0 if code is invalid or missing.
    """
    try:
        code = int(code)
    except (TypeError, ValueError):
        return 0

    if code <= 0:
        return 0

    # Convert to string to avoid ambiguity
    s = str(code)

    # Valid NAICS codes are 3 or 6 digits; take first 2 digits only
    if len(s) >= 2:
        return int(s[:2])

    return 0


def _extract_row_sectors(df: pd.DataFrame) -> List[List[int]]:
    n = len(df)
    arrs3 = [
        df[c].map(_to_int_safe).values if c in df.columns else np.zeros(n, dtype=int)
        for c in RAW_NAICS_COLS_3D
    ]
    arrs6 = [
        df[c].map(_to_int_safe).values if c in df.columns else np.zeros(n, dtype=int)
        for c in RAW_NAICS_COLS_6D
    ]

    row_sector_lists: List[List[int]] = []
    for i in range(n):
        short_codes = [a[i] for a in arrs3 if a[i] and a[i] > 0]
        long_codes = [a[i] for a in arrs6 if a[i] and a[i] > 0]
        codes = long_codes if len(long_codes) >= len(short_codes) else short_codes
        sectors = [_naics_to_2d(code) for code in codes]
        row_sector_lists.append([sector for sector in sectors if sector != 0])
    return row_sector_lists

def add_naics_coarse_features(
    df: pd.DataFrame,
    allowed_sectors: Optional[List[int]] = None,
    prefix: str = "naics_",
):
    out = df.copy()
    row_sector_lists = _extract_row_sectors(out)

    if allowed_sectors is None:
        allowed_sectors = sorted({s for sectors in row_sector_lists for s in sectors})

    new_cols: List[str] = []

    for s in allowed_sectors:
        colname = f"{prefix}sector_{s:02d}"
        out[colname] = [1.0 if s in sectors else 0.0 for sectors in row_sector_lists]
        new_cols.append(colname)

    out[f"{prefix}has_naics"] = [1.0 if len(sectors) > 0 else 0.0 for sectors in row_sector_lists]
    out[f"{prefix}n_unique_sectors_2d"] = [float(len(set(sectors))) for sectors in row_sector_lists]

    n_codes_3d = np.zeros(len(out), dtype=float)
    for c in RAW_NAICS_COLS_3D:
        if c in out.columns:
            n_codes_3d += (out[c].map(_to_int_safe).values > 0).astype(float)

    n_codes_6d = np.zeros(len(out), dtype=float)
    for c in RAW_NAICS_COLS_6D:
        if c in out.columns:
            n_codes_6d += (out[c].map(_to_int_safe).values > 0).astype(float)

    out[f"{prefix}n_codes_3d"] = n_codes_3d
    out[f"{prefix}n_codes_6d"] = n_codes_6d

    max_counts = []
    for sectors in row_sector_lists:
        if not sectors:
            max_counts.append(0.0)
        else:
            _, counts = np.unique(np.array(sectors, dtype=int), return_counts=True)
            max_counts.append(float(counts.max()))
    out[f"{prefix}max_sector_count"] = max_counts

    new_cols += [
        f"{prefix}has_naics",
        f"{prefix}n_unique_sectors_2d",
        f"{prefix}n_codes_3d",
        f"{prefix}n_codes_6d",
        f"{prefix}max_sector_count",
    ]
    return out, new_cols

def fit_naics_sector_frequency_prior(df_train: pd.DataFrame, prefix: str = "naics_"):
    priors: Dict[int, float] = {}
    sector_cols = [c for c in df_train.columns if c.startswith(f"{prefix}sector_")]
    for c in sector_cols:
        try:
            s = int(c.split("_")[-1])
        except Exception:
            continue
        priors[s] = float(pd.to_numeric(df_train[c], errors="coerce").fillna(0.0).mean())
    return priors

def add_naics_sector_frequency_features(
    df: pd.DataFrame,
    priors: Dict[int, float],
    prefix: str = "naics_",
):
    out = df.copy()
    sector_cols = [c for c in out.columns if c.startswith(f"{prefix}sector_")]
    if not sector_cols or not priors:
        out[f"{prefix}sector_freq_mean"] = 0.0
        out[f"{prefix}sector_freq_max"] = 0.0
        return out, [f"{prefix}sector_freq_mean", f"{prefix}sector_freq_max"]

    pri_vec = np.array([priors.get(int(c.split("_")[-1]), 0.0) for c in sector_cols], dtype=float)
    ind = out[sector_cols].values.astype(float)

    active = ind.sum(axis=1)
    mean_prior = np.where(active > 0, (ind * pri_vec).sum(axis=1) / active, 0.0)
    max_prior = np.where(active > 0, (ind * pri_vec).max(axis=1), 0.0)

    out[f"{prefix}sector_freq_mean"] = mean_prior
    out[f"{prefix}sector_freq_max"] = max_prior
    return out, [f"{prefix}sector_freq_mean", f"{prefix}sector_freq_max"]


class EnhancedWorkStopClassifier:
    """Enhanced classifier supporting POI features, road features, or both"""

    def __init__(self, poi_data_dir=None, road_data_dir=None, intersection_data_dir=None, intersection_feature_set='core',
                 time_data_dir=None, feature_mode='poi_time'):
        """
        Initialize the enhanced classifier

        Args:
            poi_data_dir (str, optional): Path to POI feature data directory
            road_data_dir (str, optional): Path to road feature data directory
            feature_mode (str): 'poi_only', 'road_only', or 'both' (default: 'both')
        """
        self.poi_data_dir = Path(poi_data_dir) if poi_data_dir else None
        self.road_data_dir = Path(road_data_dir) if road_data_dir else None
        self.intersection_data_dir = Path(intersection_data_dir) if intersection_data_dir else None
        self.intersection_feature_set = intersection_feature_set.lower()
        self.feature_mode = feature_mode.lower()
        self.time_data_dir= Path(time_data_dir) if time_data_dir else None

        # Validate feature mode
        valid_modes = ['poi_only', 'road_only', 'intersection_only', 'time_only', 'both', 'all', 'poi_time', 'road_time', 'intersection_time']
        if self.feature_mode not in valid_modes:
            raise ValueError(f"feature_mode must be one of {valid_modes}, got '{feature_mode}'")

        # Validate directories based on feature mode
        if self.feature_mode == 'poi_only' and not self.poi_data_dir:
            raise ValueError("poi_data_dir is required when feature_mode is 'poi_only'")
        elif self.feature_mode == 'road_only' and not self.road_data_dir:
            raise ValueError("road_data_dir is required when feature_mode is 'road_only'")
        elif self.feature_mode == 'intersection_only' and not self.intersection_data_dir:
            raise ValueError("intersection_data_dir required for intersection_only mode")
        elif self.feature_mode == 'time_only' and not self.time_data_dir:
            raise ValueError("time_data_dir required for time_only mode")
        elif self.feature_mode == 'all' and (
            not self.poi_data_dir or not self.road_data_dir or not self.intersection_data_dir
        ):
            raise ValueError("All data dirs required for 'all' feature mode")
        elif self.feature_mode == 'poi_time' and not (self.poi_data_dir or self.time_data_dir):
            raise ValueError("poi_data_dir and time_data_dir is required when feature_mode is 'poi_time'")
        elif self.feature_mode == 'road_time' and not (self.road_data_dir or self.time_data_dir):
            raise ValueError("road_data_dir and time_data_dir is required when feature_mode is 'road_time'")
        elif self.feature_mode == 'intersection_time' and not (self.intersection_data_dir or self.time_data_dir):
            raise ValueError("intersection_data_dir and time_data_dir is required when feature_mode is 'intersection_time'")

        self.data = None
        self.features = None
        self.target = None
        self.groups = None
        self.feature_names = None
        self.poi_features = None
        self.road_features = None
        self.intersection_features = None
        self.time_features = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.poi_frac_features = None
        self.model = None
        self.decision_threshold = None
        self.duration_features = None
        self.enable_naics_features = True
        self.naics_use_train_freq_priors = True

        print(f"🎯 Initialized Enhanced Classifier with feature mode: '{self.feature_mode.upper()}'")


    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba, model_name):
        """Calculate comprehensive performance metrics"""

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')

        # AUC metrics
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        # Precision-Recall AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall

        # Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2

        # Matthews Correlation Coefficient
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_accuracy,
            'mcc': mcc,
            'confusion_matrix': cm,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }

        return metrics

    def print_comprehensive_metrics(self, metrics, model_name):
        """Print comprehensive metrics in a formatted way"""

        print(f"✅ {model_name} Results:")
        print(f"   📊 Classification Metrics:")
        print(f"      Accuracy:          {metrics['accuracy']:.4f}")
        print(f"      Precision:         {metrics['precision']:.4f}")
        print(f"      Recall:            {metrics['recall']:.4f}")
        print(f"      F1-Score:          {metrics['f1_score']:.4f}")
        print(f"      Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"   📈 AUC Metrics:")
        print(f"      ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"      PR-AUC:            {metrics['pr_auc']:.4f}")
        print(f"   🎯 Additional Metrics:")
        print(f"      Specificity:       {metrics['specificity']:.4f}")
        print(f"      Sensitivity:       {metrics['sensitivity']:.4f}")
        print(f"      MCC:               {metrics['mcc']:.4f}")
        print(f"   📋 Confusion Matrix:")
        print(f"      True Positives:    {metrics['true_positives']}")
        print(f"      True Negatives:    {metrics['true_negatives']}")
        print(f"      False Positives:   {metrics['false_positives']}")
        print(f"      False Negatives:   {metrics['false_negatives']}")

    def save_model_and_metrics(self, model, model_name, metrics, training_time, base_dir=BASE_OUTPUT_DIR,
                               feature_names=None, feature_importance=None, hyperparameters=None, training_history=None,
                               test_loss=None, test_accuracy=None, extra_artifacts=None, timestamp=None):
        """Save trained model and performance metrics to files"""

        # Create timestamp
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directories
        models_dir = Path(base_dir) / "ml_results" / "saved_ml_models"
        metrics_dir = Path(base_dir) / "ml_results" / "saved_ml_performance_metrics"
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Create filenames with model_name, feature_mode, and timestamp
        model_filename = f"{model_name}_{self.feature_mode}_{timestamp}"

        try:
            # Save model
            if 'NN_' in model_name:
                # Save TensorFlow/Keras model
                model_path = models_dir / f"{model_filename}.h5"
                model.save(model_path)
                print(f"   💾 Model saved: {model_path}")
            else:
                # Save scikit-learn model with pickle
                model_path = models_dir / f"{model_filename}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"   💾 Model saved: {model_path}")

            # Prepare metrics for JSON serialization
            serializable_metrics = {"training": {}, "validation": {}, "test": {}, "keras_evaluation": {}}
            for key, value in metrics.items():
                if key == 'confusion_matrix':
                    # Convert numpy array to list
                    serializable_metrics["test"][key] = value.tolist()
                elif isinstance(value, (int, float, str, bool)):
                    serializable_metrics["test"][key] = value
                else:
                    # Convert numpy types to Python types
                    serializable_metrics["test"][key] = float(value)

            if not hyperparameters:
                try:
                    hyperparameters = model.get_params()
                except Exception as e:
                    print(e)
                    hyperparameters = "not_available"

            serializable_metrics["feature_importance"] = None

            if feature_names is not None and feature_importance is not None:
                # Random Forest and lightGBM
                if model:
                    serializable_metrics["feature_importance"] = {
                        k: float(v) for k, v in sorted(
                            zip(feature_names, feature_importance),
                            key=lambda x: x[1],
                            reverse=True
                        )
                    }
            if training_history is not None:
                history = training_history.history

                serializable_metrics["training"]["epochs_trained"] = len(history.get("loss", []))
                serializable_metrics["training"]["final_train_loss"] = float(history["loss"][-1])
                serializable_metrics["validation"]["final_val_loss"] = float(history["val_loss"][-1])
                serializable_metrics["validation"]["best_val_loss"] = float(min(history["val_loss"]))
                serializable_metrics["validation"]["best_val_epoch"] = int(
                    np.argmin(history["val_loss"]) + 1
                )
                serializable_metrics["training"]["total_train_loss"] = float(np.sum(history["loss"]))
                serializable_metrics["validation"]["total_val_loss"] = float(np.sum(history["val_loss"]))
                serializable_metrics["keras_evaluation"]["test_loss"] = float(test_loss)
                serializable_metrics["keras_evaluation"]["test_accuracy"] = float(test_accuracy)

                # Learning rate tracking
                lr_history = history.get("lr") or history.get("learning_rate")
                if lr_history:
                    serializable_metrics["training"]["initial_learning_rate"] = float(lr_history[0])
                    serializable_metrics["training"]["final_learning_rate"] = float(lr_history[-1])
                else:
                    try:
                        lr = model.optimizer.learning_rate
                        if callable(lr):
                            lr = lr(model.optimizer.iterations)
                        serializable_metrics["training"]["initial_learning_rate"] = float(lr)
                        serializable_metrics["training"]["final_learning_rate"] = float(lr)
                    except Exception:
                        serializable_metrics["training"]["initial_learning_rate"] = None
                        serializable_metrics["training"]["final_learning_rate"] = None

            # Add additional metadata
            metrics_data = {
                'model_name': model_name,
                'feature_mode': self.feature_mode,
                'timestamp': timestamp,
                'training_time_seconds': training_time,
                'dataset_size': len(self.data) if self.data is not None else 0,
                'num_features': len(self.feature_names) if self.feature_names else 0,
                "hyperparameters": hyperparameters,
                'metrics': serializable_metrics
            }
            if extra_artifacts is not None:
                metrics_data["artifacts"] = extra_artifacts

            # Save metrics
            metrics_path = metrics_dir / f"{model_filename}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            print(f"   📊 Metrics saved: {metrics_path}")

            return model_path, metrics_path

        except Exception as e:
            print(f"   ❌ Error saving model/metrics: {e}")
            return None, None

    def load_and_merge_data(self, num_files=NUM_FILES_EACH_FOLDER, save_enhanced_feature_files=SAVE_ENHANCED_FILES):
        """Load data based on feature mode (POI only, road only, or both)"""

        print(f"🔄 Loading data for feature mode: '{self.feature_mode.upper()}'...")

        if self.feature_mode == 'poi_only':
            return self._load_poi_only_data(num_files)
        elif self.feature_mode == 'road_only':
            return self._load_road_only_data(num_files)
        elif self.feature_mode == 'intersection_only':
            return self._load_intersection_only_data(num_files)
        elif self.feature_mode == 'time_only':
            return self._load_time_only_data(num_files)
        # elif self.feature_mode == 'both':  # both
        #     return self._load_combined_data(num_files)
        elif self.feature_mode == 'all':
            return self._load_all_data(
                num_files,
                save_enhanced_feature_files=save_enhanced_feature_files,
            )
        elif self.feature_mode == 'poi_time':
            return self._load_combined_data(num_files, self.poi_data_dir, self.time_data_dir,'with_poi_features', 'with_time_features')
        elif self.feature_mode == 'road_time':
            return self._load_combined_data(num_files, self.road_data_dir, self.time_data_dir, 'with_road_features', 'with_time_features')
        elif self.feature_mode == 'intersection_time':
            return self._load_combined_data(num_files, self.intersection_data_dir, self.time_data_dir,'with_intersection_features', 'with_time_features')

    def _load_poi_only_data(self, num_files):
        """Load POI feature data only"""

        poi_files = list(self.poi_data_dir.glob("*_with_poi_features.csv"))
        print(f"Found {len(poi_files)} POI files")

        if not poi_files:
            raise FileNotFoundError(f"No POI files found in {self.poi_data_dir}")

        # Handle num_files=None (load all files)
        if num_files is None:
            files_to_load = poi_files
            total_files = len(poi_files)
        else:
            files_to_load = poi_files[:num_files]
            total_files = min(num_files, len(poi_files))

        # Load POI data
        dataframes = []
        for i, poi_file in enumerate(files_to_load, 1):
            print(f"Loading POI file {i}/{total_files}: {poi_file.name}")
            df = pd.read_csv(poi_file)
            dataframes.append(df)

        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(self.data)} total records with {self.data.shape[1]} columns (POI only)")

        return self.data

    def _load_road_only_data(self, num_files):
        """Load road feature data only"""

        road_files = list(self.road_data_dir.glob("*_with_road_features.csv"))
        print(f"Found {len(road_files)} road files")

        if not road_files:
            raise FileNotFoundError(f"No road files found in {self.road_data_dir}")

        # Handle num_files=None (load all files)
        if num_files is None:
            files_to_load = road_files
            total_files = len(road_files)
        else:
            files_to_load = road_files[:num_files]
            total_files = min(num_files, len(road_files))

        # Load road data
        dataframes = []
        for i, road_file in enumerate(files_to_load, 1):
            print(f"Loading road file {i}/{total_files}: {road_file.name}")
            df = pd.read_csv(road_file)
            dataframes.append(df)

        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(self.data)} total records with {self.data.shape[1]} columns (road only)")

        return self.data

    def _load_intersection_only_data(self, num_files):
        """Load road feature data only"""

        intersection_files = list(self.intersection_data_dir.glob("*_with_intersection_features.csv"))
        print(f"Found {len(intersection_files)} road files")

        if not intersection_files:
            raise FileNotFoundError(f"No road files found in {self.intersection_data_dir}")

        # Handle num_files=None (load all files)
        if num_files is None:
            files_to_load = intersection_files
            total_files = len(intersection_files)
        else:
            files_to_load = intersection_files[:num_files]
            total_files = min(num_files, len(intersection_files))

        # Load road data
        dataframes = []
        for i, intersection_file in enumerate(files_to_load, 1):
            print(f"Loading road file {i}/{total_files}: {intersection_file.name}")
            df = pd.read_csv(intersection_file)
            dataframes.append(df)

        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(self.data)} total records with {self.data.shape[1]} columns (intersection only)")

        return self.data

    def _load_time_only_data(self, num_files):
        """Load POI feature data only"""

        time_files = list(self.time_data_dir.glob("*_with_time_features.csv"))
        print(f"Found {len(time_files)} POI files")

        if not time_files:
            raise FileNotFoundError(f"No POI files found in {self.time_data_dir}")

        # Handle num_files=None (load all files)
        if num_files is None:
            files_to_load = time_files
            total_files = len(time_files)
        else:
            files_to_load = time_files[:num_files]
            total_files = min(num_files, len(time_files))

        # Load POI data
        dataframes = []
        for i, time_file in enumerate(files_to_load, 1):
            print(f"Loading POI file {i}/{total_files}: {time_file.name}")
            df = pd.read_csv(time_file)
            dataframes.append(df)

        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(self.data)} total records with {self.data.shape[1]} columns (POI only)")

        return self.data

    def _load_combined_data(self, num_files, data_dir1, data_dir2, data_pattern1, data_pattern2):
        """Load and merge both POI and road feature data"""

        # Find matching files
        # poi_files = list(self.poi_data_dir.glob("*_with_poi_features.csv"))
        # road_files = list(self.road_data_dir.glob("*_with_road_features.csv"))
        poi_files = list(data_dir1.glob("*_{}.csv".format(data_pattern1)))
        road_files = list(data_dir2.glob("*_{}.csv".format(data_pattern2)))

        print(f"Found {len(poi_files)} {data_dir1} files and {len(road_files)} {data_dir2} files")

        # Match files by base name
        matched_files = []
        for poi_file in poi_files:
            base_name = poi_file.name.replace('_{}.csv'.format(data_pattern1), '')
            road_file = data_dir2 / f"{base_name}_{data_pattern2}.csv"

            if road_file.exists():
                matched_files.append((poi_file, road_file))

        print(f"Found {len(matched_files)} matching file pairs")

        if len(matched_files) == 0:
            raise FileNotFoundError("No matching POI and road feature files found!")

        # Handle num_files=None (load all files)
        if num_files is None:
            files_to_load = matched_files
            total_files = len(matched_files)
        else:
            files_to_load = matched_files[:num_files]
            total_files = min(num_files, len(matched_files))

        # Load and merge data
        dataframes = []
        for i, (poi_file, road_file) in enumerate(files_to_load, 1):
            print(f"Loading pair {i}/{total_files}: {poi_file.name}")

            # Load POI data
            poi_df = pd.read_csv(poi_file)

            # Load road data
            road_df = pd.read_csv(road_file)

            # Merge on common columns (assuming same order and content)
            if len(poi_df) != len(road_df):
                print(f"   Warning: {data_dir1} file has {len(poi_df)} rows, {data_dir2} file has {len(road_df)} rows")
                min_len = min(len(poi_df), len(road_df))
                poi_df = poi_df.iloc[:min_len]
                road_df = road_df.iloc[:min_len]

            # Get road feature columns (exclude common columns)
            common_cols = set(poi_df.columns) & set(road_df.columns)
            road_feature_cols = [col for col in road_df.columns if col not in common_cols]

            # Merge dataframes
            merged_df = poi_df.copy()
            for col in road_feature_cols:
                merged_df[col] = road_df[col]

            dataframes.append(merged_df)

        # Combine all dataframes
        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(self.data)} total records with {self.data.shape[1]} columns ({data_dir1} + {data_dir2})")

        return self.data


    def _enhanced_feature_export_columns(self):
        """Columns written back to per-file enhanced feature CSVs."""
        return [
            col for col in self.data.columns
            if (
                col.startswith("poi_frac_")
                or col.startswith("durbin_")
                or col == "log_stop_duration_s"
                or col.startswith("int_logdur_")
                or col.startswith("int_durbin_")
                or col.startswith("naics_")
            )
        ]

    def _enhanced_feature_export_name(self, col):
        if col.startswith("int_durbin_"):
            return col.replace("int_durbin_", "durbin_", 1)
        return col

    def _save_enhanced_feature_files(
        self,
        file_slices,
        raw_data_dir=RAW_STOP_RECORDS_DIR,
        output_dir=ENHANCED_FEATURES_OUTPUT_DIR,
    ):
        """Save original Kaggle rows plus newly engineered features per source file."""
        raw_data_dir = Path(raw_data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        enhanced_cols = self._enhanced_feature_export_columns()
        if not enhanced_cols:
            print("Warning: no enhanced feature columns found to export")
            return

        saved_count = 0
        for base_name, start_idx, end_idx in file_slices:
            raw_file = raw_data_dir / f"{base_name}.csv"
            if raw_file.exists():
                out_df = pd.read_csv(raw_file)
            else:
                print(f"Warning: raw Kaggle file missing, using merged rows instead: {raw_file}")
                out_df = self.data.iloc[start_idx:end_idx].copy()

            feature_df = self.data.iloc[start_idx:end_idx][enhanced_cols].reset_index(drop=True)
            out_df = out_df.reset_index(drop=True)

            if len(out_df) != len(feature_df):
                print(
                    f"Warning: row count mismatch for {base_name}: "
                    f"raw={len(out_df)}, enhanced={len(feature_df)}; missing feature rows will be blank"
                )
                feature_df = feature_df.reindex(range(len(out_df)))

            for col in enhanced_cols:
                out_df[self._enhanced_feature_export_name(col)] = feature_df[col]

            output_file = output_dir / f"{base_name}_with_enhanced_features.csv"
            out_df.to_csv(output_file, index=False)
            saved_count += 1

        print(f"✅ Saved {saved_count} enhanced feature files to {output_dir}")


    def _load_all_data(self, num_files, save_enhanced_feature_files=SAVE_ENHANCED_FILES):
        """Load and merge both POI and road feature data"""

        # Find matching files
        poi_files = list(self.poi_data_dir.glob("*_with_poi_features.csv"))
        road_files = list(self.road_data_dir.glob("*_with_road_features.csv"))
        intersection_files = list(self.intersection_data_dir.glob("*_with_intersection_features.csv"))
        time_files = list(self.time_data_dir.glob("*_with_time_features.csv"))

        print(f"Found {len(poi_files)} POI files and {len(road_files)} road files and {len(intersection_files)} intersection files and {len(time_files)} time files")

        # Match files by base name
        matched_files = []
        for poi_file in poi_files:
            base_name = poi_file.name.replace('_with_poi_features.csv', '')
            road_file = self.road_data_dir / f"{base_name}_with_road_features.csv"
            intersection_file = self.intersection_data_dir / f"{base_name}_with_intersection_features.csv"
            time_file = self.time_data_dir / f"{base_name}_with_time_features.csv"

            if road_file.exists() and intersection_file.exists() and time_file.exists():
                matched_files.append((base_name, poi_file, road_file, intersection_file, time_file))

        print(f"Found {len(matched_files)} matching file pairs")

        if len(matched_files) == 0:
            raise FileNotFoundError("No matching POI and road feature files found!")

        # Handle num_files=None (load all files)
        if num_files is None:
            files_to_load = matched_files
            total_files = len(matched_files)
        else:
            files_to_load = matched_files[:num_files]
            total_files = min(num_files, len(matched_files))

        # Load and merge data
        dataframes = []
        file_slices = []
        next_start_idx = 0
        for i, (base_name, poi_file, road_file, intersection_file, time_file) in enumerate(files_to_load, 1):
            print(f"Loading pair {i}/{total_files}: {poi_file.name}")

            # Load POI data
            poi_df = pd.read_csv(poi_file)

            # Load road data
            road_df = pd.read_csv(road_file)

            intersection_df = pd.read_csv(intersection_file)

            time_df = pd.read_csv(time_file)

            # Merge on common columns (assuming same order and content)
            if len(poi_df) != len(road_df):
                print(f"   Warning: POI file has {len(poi_df)} rows, road file has {len(road_df)} rows")
                min_len = min(len(poi_df), len(road_df))
                poi_df = poi_df.iloc[:min_len]
                road_df = road_df.iloc[:min_len]
                intersection_df = intersection_df.iloc[:min_len]
                time_df = time_df.iloc[:min_len]

            # Get road feature columns (exclude common columns)
            common_cols = set(poi_df.columns) & set(road_df.columns) & set(intersection_df.columns) & set(time_df.columns)
            road_feature_cols = [col for col in road_df.columns if col not in common_cols]
            intersection_feature_cols = [col for col in intersection_df.columns if col not in common_cols]
            time_feature_cols = [col for col in time_df.columns if col not in common_cols]

            # Merge dataframes
            merged_df = poi_df.copy()
            for col in road_feature_cols:
                merged_df[col] = road_df[col]

            for col in intersection_feature_cols:
                merged_df[col] = intersection_df[col]

            for col in time_feature_cols:
                merged_df[col] = time_df[col]

            dataframes.append(merged_df)
            row_count = len(merged_df)
            file_slices.append((base_name, next_start_idx, next_start_idx + row_count))
            next_start_idx += row_count

        # Combine all dataframes
        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(self.data)} total records with {self.data.shape[1]} columns (POI + road + intersection + time)")

        poi_fraction_base_cols = [
            "Restaurants, Fast Food and Cafes",
            "Shopping and Retail",
            "Transportation and Transit",
            "Parking",
            "Gas Stations",
        ]

        self.data, new_poi_frac_cols = add_poi_fraction_features(
            self.data,
            poi_category_cols=poi_fraction_base_cols,
            total_pois_col="total_pois",
            prefix="poi_frac_",
            add_commercial_frac=True,
        )
        print(f"✅ Added {len(new_poi_frac_cols)} POI fraction features: {new_poi_frac_cols[:5]}{'...' if len(new_poi_frac_cols)>5 else ''}")

        # ---- Phase 3 NEXT: duration-context interactions ----
        interaction_poi_fracs = [
            "poi_frac_commercial_frac",
            "poi_frac_Restaurants, Fast Food and Cafes",
            "poi_frac_Shopping and Retail",
            "poi_frac_Transportation and Transit",
            "poi_frac_Parking",
            "poi_frac_Gas Stations",
        ]
        self.data, new_int_cols = add_duration_context_interactions(
            self.data,
            duration_col="stop_duration_s",
            poi_frac_cols=interaction_poi_fracs,
            include_duration_bins=True,
            prefix="int_",
        )

        print(f"✅ Added {len(new_int_cols)} duration×context interaction features")

        # ---- Phase 3 NAICS (coarse) - SAFE to do here (no priors, no leakage) ----
        if getattr(self, "enable_naics_features", False):
            # Add multi-hot sectors + diversity features to self.data
            self.data, naics_cols = add_naics_coarse_features(
                self.data,
                allowed_sectors=None,   # OK here because it's still pure transform
                prefix="naics_"
            )
            print(f"✅ Added {len(naics_cols)} NAICS coarse features (no priors yet)")

        if save_enhanced_feature_files:
            self._save_enhanced_feature_files(file_slices)
        else:
            print("Skipping enhanced feature file export")

        return self.data


    def prepare_features(self):
        """Prepare features for ML training based on feature mode"""

        print(f"\n🔧 Preparing features for mode: '{self.feature_mode.upper()}'...")

        # Define columns to exclude
        exclude_columns = [
            'unit_id', 'stop_start_time_local', 'stop_end_time_local',
            'is_work_stop', 'distance_to_work_stop', 'planned_arrival_time_local',
            'planned_departure_time_local',
            'naics1', 'naics2', 'naics3','naics1_long', 'naics2_long', 'naics3_long',
            'POIs', 'Stop Road Info', 'Nearest Intersections'
        ]

        # Get all potential feature columns
        all_feature_columns = [col for col in self.data.columns if col not in exclude_columns]

        # Identify POI and road features

        poi_features = ['Accommodation', 'Automative Services', 'Education and Learning', 'Emergency and Safety',
                        'Entertainment and Recreation', 'Events and Venues', 'Financial Services',
                        'Gas Stations', 'Government and Public Services', 'Healthcare and Medical',
                        'Industrial and Manufacturing', 'Other',
                        'Parking', 'Personal Care and Beauty', 'Pharmacy', 'Post', 'Professional and Business Services',
                        'Religious and Spiritual', 'Residential', 'Restaurants, Fast Food and Cafes',
                        'Shopping and Retail', 'Tourism and Attractions', 'Transportation and Transit',
                        'Waste and Recycling', 'total_pois']

        road_features = ['busway_count', 'cycleway_count', 'minor_road_count', 'motorway_count', 'pedestrian_count',
                         'primary_road_count', 'residential_road_count', 'rest_area_count', 'road_construction_count',
                         'secondary_road_count', 'service_road_count', 'tertiary_road_count', 'trunk_count',
                         'has_speed_limit', 'max_speed_numeric', 'low_speed_roads', 'medium_speed_roads', 'high_speed_roads',
                         'total_lanes', 'max_lanes', 'has_multi_lane', 'has_road_features',
                         'closest_road_distance', 'avg_road_distance', 'total_roads_count']


        intersection_keywords = ['intersection', 'sign', 'crossing', 'barrier', 'railway', 'access']
        intersection_features = [col for col in all_feature_columns if
                                 any(k in col.lower() for k in intersection_keywords)]

        core_intersection_features = ['has_intersections', 'closest_intersection_distance', 'average_intersection_distance',
                                      'total_intersections', 'nearest_intersection_sign_distance',
                                      'intersection_has_speed_limit', 'intersection_max_speed_numeric',
                                      'intersection_low_speed_roads', 'intersection_medium_speed_roads',
                                      'intersection_high_speed_roads', 'intersection_has_multi_lane',
                                      'total_signs', 'total_stop_sign', 'total_give_way_sign',
                                      'has_crossing', 'has_railway', 'has_barrier']

        time_features = ['stop_duration_s', 'stop_start_time_local_seconds', 'stop_start_time_local_sin', 'stop_start_time_local_cos',
                         'stop_end_time_local_seconds', 'stop_end_time_local_sin', 'stop_end_time_local_cos', 'is_weekend']

        if self.intersection_feature_set == 'core':
            intersection_features = [f for f in intersection_features if f in core_intersection_features]

        poi_frac_features = [c for c in all_feature_columns if c.startswith("poi_frac_")]
        duration_features = [c for c in all_feature_columns if c.startswith("int_")]
        duration_features.append('log_stop_duration_s')

        naics_features = [c for c in all_feature_columns if c.startswith("naics_")]

        # Select features based on mode
        if self.feature_mode == 'poi_only':
            selected_features = poi_features
            print(f"Selected {len(selected_features)} POI features")
        elif self.feature_mode == 'road_only':
            selected_features = road_features
            print(f"Selected {len(selected_features)} road features")
        elif self.feature_mode == 'intersection_only':
            selected_features = intersection_features
        elif self.feature_mode == 'time_only':
            selected_features = time_features
        elif self.feature_mode == 'all':
            selected_features = list(set(poi_features + poi_frac_features + road_features + intersection_features + time_features + duration_features))
            if getattr(self, "enable_naics_features", False):
                selected_features = list(set(selected_features + naics_features))
        elif self.feature_mode == 'poi_time':
            selected_features = list(set(poi_features + time_features))
        elif self.feature_mode == 'road_time':
            selected_features = list(set(road_features + time_features))
        elif self.feature_mode == 'intersection_time':
            selected_features = list(set(intersection_features + time_features))

        # Store feature lists for analysis
        self.poi_features = poi_features
        self.road_features = road_features
        self.intersection_features = intersection_features
        self.time_features = time_features
        self.poi_frac_features = poi_frac_features
        self.duration_features = duration_features

        # Prepare data
        self.features = self.data[selected_features].fillna(-1)
        self.target = self.data['is_work_stop']
        self.groups = self.data['unit_id']
        #self.groups = self.data['naics1_long']
        self.feature_names = list(self.features.columns)

        print(f"\nDataset info:")
        print(f"   Samples: {len(self.features)}")
        print(f"   Features: {self.features.shape[1]}")
        print(f"   Work stops: {(self.target == 1).sum()}")
        print(f"   Non-work stops: {(self.target == 0).sum()}")
        print(f"   Unique vehicles: {self.groups.nunique()}")
        print(f"   POI frac features added: {len(self.poi_frac_features)}")
        print(f"   Duration features added: {len(self.duration_features)}")

        return poi_features, road_features, intersection_features, time_features

    def split_data(self, test_size=0.15, val_size=0.15, random_state=RANDOM_STATE):
        """
        Group-aware split into train / validation / test
        """

        print("\n📊 Splitting data (group-aware train/val/test)...")

        # 1️⃣ Train+Val vs Test
        gss_test = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state
        )

        train_val_idx, test_idx = next(
            gss_test.split(self.features, self.target, groups=self.groups)
        )

        X_train_val = self.features.iloc[train_val_idx]
        y_train_val = self.target.iloc[train_val_idx]
        groups_train_val = self.groups.iloc[train_val_idx]

        X_test = self.features.iloc[test_idx]
        y_test = self.target.iloc[test_idx]

        # 2️⃣ Train vs Validation (from train_val)
        val_ratio_adjusted = val_size / (1 - test_size)

        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=val_ratio_adjusted,
            random_state=random_state
        )

        train_idx, val_idx = next(
            gss_val.split(X_train_val, y_train_val, groups=groups_train_val)
        )

        X_train = X_train_val.iloc[train_idx]
        y_train = y_train_val.iloc[train_idx]

        X_val = X_train_val.iloc[val_idx]
        y_val = y_train_val.iloc[val_idx]

        print(f"   🟢 Train samples: {len(X_train)}")
        print(f"   🟡 Val samples:   {len(X_val)}")
        print(f"   🔵 Test samples:  {len(X_test)}")

        # Store splits and indices for later diagnostics (no behavior change)
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        # Keep original row indices into self.data for traceability
        self.idx_train = X_train.index.to_numpy()
        self.idx_val = X_val.index.to_numpy()
        self.idx_test = X_test.index.to_numpy()

        return X_train, X_val, X_test, y_train, y_val, y_test

    def add_naics_priors_post_split(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        prefix: str = "naics_"
        ):
        """
        Adds TRAIN-only frequency prior features for NAICS sectors.
        Requires naics_sector_XX columns already exist in X_* (created in _load_all_data()).
        """
        if not getattr(self, "naics_use_train_freq_priors", True):
            return X_train, X_val, X_test

        priors = fit_naics_sector_frequency_prior(X_train, prefix=prefix)

        X_train2, _ = add_naics_sector_frequency_features(X_train, priors, prefix=prefix)
        X_val2, _ = add_naics_sector_frequency_features(X_val, priors, prefix=prefix)
        X_test2, _ = add_naics_sector_frequency_features(X_test, priors, prefix=prefix)

        return X_train2, X_val2, X_test2

    def load_trained_model(self, model_pkl_path: str):
        """Load a trained model from a pickle file and assign to self.model."""
        import pickle
        from pathlib import Path

        p = Path(model_pkl_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

        with p.open("rb") as f:
            obj = pickle.load(f)

        # Common patterns: model itself, dict wrapper, or pipeline-like object
        model = obj
        if isinstance(obj, dict):
            for k in ["model", "clf", "estimator", "lgbm", "booster"]:
                if k in obj:
                    model = obj[k]
                    break

        self.model = model
        self.model_pkl_path = str(p)
        print(f"✅ Loaded model from {p} (type={type(model)})")
        return model

    def load_decision_threshold_from_metrics(self, metrics_json_path: str) -> float:
        """Load decision threshold from a saved metrics JSON and store to self.decision_threshold."""
        import json
        from pathlib import Path

        p = Path(metrics_json_path)
        if not p.exists():
            raise FileNotFoundError(f"Metrics JSON not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        thr = None
        if isinstance(metrics, dict):
            if "hyperparameters" in metrics and isinstance(metrics["hyperparameters"], dict):
                thr = metrics["hyperparameters"].get("decision_threshold", None)
            if thr is None:
                thr = metrics.get("decision_threshold", None)

        if thr is None:
            raise KeyError(
                "Could not find decision threshold in metrics JSON. "
                "Expected metrics['hyperparameters']['decision_threshold'] or metrics['decision_threshold']"
            )

        self.decision_threshold = float(thr)
        self.metrics_json_path = str(p)
        print(f"✅ Loaded decision_threshold={self.decision_threshold:.6f} from {p}")
        return self.decision_threshold

    def prepare_splits_only(self, seed: int, test_size: float = 0.15, val_size: float = 0.15):
        """Load data, prepare features, and create train/val/test splits (no training, no tuning)."""
        # Ensure data is loaded
        if not hasattr(self, "data") or self.data is None:
            # reuse your existing loader
            self.load_and_merge_data()
        # Ensure features are prepared
        self.prepare_features()
        # Split and store as attributes (split_data now stores attributes)
        return self.split_data(test_size=test_size, val_size=val_size, random_state=seed)

    def run_fp_diagnosis(
        self,
        seed: int,
        out_dir: str,
        split: str = "test",
        top_k_features: int = 20,
    ) -> dict:
        """Runs FP vs TN diagnostics at the current decision threshold on VAL or TEST.

        Assumes:
          - self.model is set (trained or loaded)
          - self.decision_threshold is set (tuned or loaded from metrics)
          - split_data / prepare_splits_only has been run so X_val/X_test exist
        """
        if split not in {"test", "val"}:
            raise ValueError("split must be 'test' or 'val'")

        # Retrieve split data
        if split == "test":
            if not hasattr(self, "X_test"):
                raise RuntimeError("X_test not found; run split_data(...) or prepare_splits_only(...) first.")
            X_split = self.X_test
            y_split = self.y_test
            idx_split = getattr(self, "idx_test", None)
        else:
            if not hasattr(self, "X_val"):
                raise RuntimeError("X_val not found; run split_data(...) or prepare_splits_only(...) first.")
            X_split = self.X_val
            y_split = self.y_val
            idx_split = getattr(self, "idx_val", None)

        if idx_split is None:
            raise RuntimeError("Split indices not found; ensure split_data stored idx_train/idx_val/idx_test.")

        model = self.model
        if model is None:
            raise RuntimeError("Model is None; call load_trained_model(...) or train_lightgbm(...) first.")

        if not hasattr(self, "decision_threshold"):
            raise RuntimeError("decision_threshold missing; call load_decision_threshold_from_metrics(...) or run tuning.")

        thr = float(self.decision_threshold)

        # --- Align X to model (sanitize + reorder) ---
        def _sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.columns = df.columns.str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)
            return df

        def _expected_feature_names(mdl):
            # sklearn LGBMClassifier
            if hasattr(mdl, "feature_name_"):
                try:
                    return list(mdl.feature_name_)
                except Exception:
                    pass
            # sklearn wrapper sometimes stores booster_
            if hasattr(mdl, "booster_") and hasattr(mdl.booster_, "feature_name"):
                try:
                    return list(mdl.booster_.feature_name())
                except Exception:
                    pass
            # lightgbm.Booster
            if hasattr(mdl, "feature_name") and callable(getattr(mdl, "feature_name")):
                try:
                    return list(mdl.feature_name())
                except Exception:
                    pass
            if hasattr(mdl, "feature_names_"):
                try:
                    return list(mdl.feature_names_)
                except Exception:
                    pass
            return None

        def _align_X_to_model(df: pd.DataFrame, mdl) -> pd.DataFrame:
            Xs = _sanitize_cols(df)
            expected = _expected_feature_names(mdl)
            if expected is None:
                print("⚠️  Could not determine model feature names; using sanitized X_split as-is.")
                return Xs
            missing = [c for c in expected if c not in Xs.columns]
            if missing:
                raise RuntimeError(
                    f"X_split missing {len(missing)} features expected by model. First 30: {missing[:30]}"
                )
            extra = [c for c in Xs.columns if c not in expected]
            if extra:
                print(f"ℹ️  Dropping {len(extra)} extra columns not used by model.")
            return Xs[expected]

        X_pred = _align_X_to_model(X_split, model)
        print(f"🔎 FP diagnosis using model type={type(model)} | n_features={X_pred.shape[1]} | threshold={thr:.6f}")
        print(f"   First 10 features: {list(X_pred.columns[:10])}")

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_pred)[:, 1]
        else:
            y_prob = model.predict(X_pred)
        y_prob = np.asarray(y_prob, dtype=float)

        y_true = y_split.values if hasattr(y_split, "values") else np.asarray(y_split)
        y_pred = (y_prob >= thr).astype(int)

        # Build evaluation DF using the exact columns the model saw
        df_eval = X_pred.copy()
        df_eval["y_true"] = y_true
        df_eval["y_prob"] = y_prob
        df_eval["y_pred"] = y_pred

        error_type = np.full(len(df_eval), "TN", dtype=object)
        error_type[(y_true == 1) & (y_pred == 1)] = "TP"
        error_type[(y_true == 1) & (y_pred == 0)] = "FN"
        error_type[(y_true == 0) & (y_pred == 1)] = "FP"
        df_eval["error_type"] = error_type
        df_eval["row_index"] = np.asarray(idx_split)

        if hasattr(self, "data") and self.data is not None and "unit_id" in self.data.columns:
            try:
                df_eval["unit_id"] = self.data.loc[idx_split, "unit_id"].values
            except Exception:
                pass

        out_path = Path(out_dir)
        os.makedirs(out_path, exist_ok=True)

        artifacts = {}

        # Save predictions
        pred_path = out_path / f"seed{seed}_{split}_predictions.parquet"
        try:
            df_eval.to_parquet(pred_path, index=False)
            artifacts["predictions"] = str(pred_path)
        except Exception:
            pred_csv = out_path / f"seed{seed}_{split}_predictions.csv"
            df_eval.to_csv(pred_csv, index=False)
            artifacts["predictions"] = str(pred_csv)

        # Save FP rows
        fp_rows = df_eval[(df_eval["y_true"] == 0) & (df_eval["y_pred"] == 1)]
        fp_path = out_path / f"seed{seed}_{split}_fp_rows.csv"
        fp_rows.to_csv(fp_path, index=False)
        artifacts["fp_rows"] = str(fp_path)

        # --- Slice helpers that work with sanitized column names ---
        def _san(name: str) -> str:
            return _re.sub(r'[^0-9a-zA-Z_]+', '_', name)

        def _col(name: str):
            if name in df_eval.columns:
                return name
            s = _san(name)
            if s in df_eval.columns:
                return s
            return None

        slices = []
        total_fp = int(fp_rows.shape[0])

        def add_slice(slice_name, series, bins, labels):
            if series is None:
                return
            cats = pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)
            df_tmp = df_eval[["y_true", "y_pred"]].copy()
            df_tmp["_slice"] = cats
            for label in labels:
                mask = df_tmp["_slice"] == label
                nonwork = df_tmp[mask & (df_tmp["y_true"] == 0)]
                fp = df_tmp[mask & (df_tmp["y_true"] == 0) & (df_tmp["y_pred"] == 1)]
                n_nonwork = int(nonwork.shape[0])
                n_fp = int(fp.shape[0])
                fpr = (n_fp / n_nonwork) if n_nonwork > 0 else 0.0
                fp_share = (n_fp / total_fp) if total_fp > 0 else 0.0
                slices.append({
                    "slice": slice_name,
                    "bucket": str(label),
                    "n_nonwork": n_nonwork,
                    "fp": n_fp,
                    "fpr": fpr,
                    "fp_share": fp_share,
                })

        def add_presence_slice(feature_name):
            col = _col(feature_name)
            if col is None:
                print(f"Warning: feature '{feature_name}' not found; skipping presence slice")
                return
            mask = df_eval[col] > 0
            nonwork = df_eval[mask & (df_eval["y_true"] == 0)]
            fp = df_eval[mask & (df_eval["y_true"] == 0) & (df_eval["y_pred"] == 1)]
            n_nonwork = int(nonwork.shape[0])
            n_fp = int(fp.shape[0])
            fpr = (n_fp / n_nonwork) if n_nonwork > 0 else 0.0
            fp_share = (n_fp / total_fp) if total_fp > 0 else 0.0
            slices.append({
                "slice": f"poi_presence:{feature_name}",
                "bucket": ">0",
                "n_nonwork": n_nonwork,
                "fp": n_fp,
                "fpr": fpr,
                "fp_share": fp_share,
            })

        # numeric slices
        col = _col("stop_duration_s")
        if col:
            add_slice("stop_duration_s", df_eval[col], [0, 120, 600, 1800, float("inf")], ["0-120", "120-600", "600-1800", "1800+"])
        else:
            print("Warning: stop_duration_s not found; skipping duration bins")

        col = _col("stop_start_time_local_seconds")
        if col:
            hours = (df_eval[col] // 3600) % 24
            add_slice("start_hour", hours, [0, 6, 9, 12, 15, 18, 22, 24], ["0-5", "6-8", "9-11", "12-14", "15-17", "18-21", "22-23"])
        else:
            print("Warning: stop_start_time_local_seconds not found; skipping hour bins")

        col = _col("is_weekend")
        if col:
            add_slice("is_weekend", df_eval[col], [-0.5, 0.5, 1.5], ["0", "1"])
        else:
            print("Warning: is_weekend not found; skipping weekend bins")

        col = _col("total_pois")
        if col:
            add_slice("total_pois", df_eval[col], [0, 3, 11, 31, float("inf")], ["0-2", "3-10", "11-30", "31+"])
        else:
            print("Warning: total_pois not found; skipping total_pois bins")

        col = _col("total_signs")
        if col:
            add_slice("total_signs", df_eval[col], [0, 1, 6, 21, float("inf")], ["0", "1-5", "6-20", "21+"])
        else:
            print("Warning: total_signs not found; skipping total_signs bins")

        col = _col("total_intersections")
        if col:
            add_slice("total_intersections", df_eval[col], [0, 1, 4, 11, float("inf")], ["0", "1-3", "4-10", "11+"])
        else:
            print("Warning: total_intersections not found; skipping total_intersections bins")

        col = _col("closest_intersection_distance")
        if col:
            add_slice("closest_intersection_distance", df_eval[col], [0, 25, 75, 150, float("inf")], ["0-25", "25-75", "75-150", "150+"])
        else:
            print("Warning: closest_intersection_distance not found; skipping intersection distance bins")

        col = _col("closest_road_distance")
        if col:
            add_slice("closest_road_distance", df_eval[col], [0, 25, 75, 150, float("inf")], ["0-25", "25-75", "75-150", "150+"])
        else:
            print("Warning: closest_road_distance not found; skipping road distance bins")

        poi_presence_features = [
            "Restaurants, Fast Food and Cafes",
            "Shopping and Retail",
            "Transportation and Transit",
            "Parking",
            "Gas Stations",
        ]
        for feature_name in poi_presence_features:
            add_presence_slice(feature_name)

        slices_df = pd.DataFrame(slices)
        slices_path = out_path / f"seed{seed}_{split}_fp_slices.csv"
        slices_df.to_csv(slices_path, index=False)
        artifacts["fp_slices"] = str(slices_path)

        # FP vs TN feature separation
        candidate_features = [
            "stop_duration_s",
            "stop_start_time_local_seconds",
            "stop_end_time_local_seconds",
            "total_pois",
            "total_signs",
            "total_intersections",
            "closest_road_distance",
            "avg_road_distance",
            "closest_intersection_distance",
            "average_intersection_distance",
            "nearest_intersection_sign_distance",
        ] + poi_presence_features

        fp_mask = (df_eval["y_true"] == 0) & (df_eval["y_pred"] == 1)
        tn_mask = (df_eval["y_true"] == 0) & (df_eval["y_pred"] == 0)

        try:
            from scipy.stats import ks_2samp
        except Exception:
            ks_2samp = None

        sep_rows = []
        for feat in candidate_features:
            col = _col(feat)
            if col is None:
                continue
            series = pd.to_numeric(df_eval[col], errors="coerce")
            fp_vals = series[fp_mask].dropna().values
            tn_vals = series[tn_mask].dropna().values
            if len(fp_vals) == 0 or len(tn_vals) == 0:
                continue

            median_fp = float(np.median(fp_vals))
            median_tn = float(np.median(tn_vals))
            delta_median = median_fp - median_tn

            mean_fp = float(np.mean(fp_vals))
            mean_tn = float(np.mean(tn_vals))
            std_fp = float(np.std(fp_vals, ddof=1)) if len(fp_vals) > 1 else 0.0
            std_tn = float(np.std(tn_vals, ddof=1)) if len(tn_vals) > 1 else 0.0

            pooled = 0.0
            if len(fp_vals) > 1 and len(tn_vals) > 1:
                pooled = np.sqrt(((len(fp_vals)-1)*std_fp**2 + (len(tn_vals)-1)*std_tn**2) / (len(fp_vals)+len(tn_vals)-2))
            smd = (mean_fp - mean_tn) / pooled if pooled > 0 else 0.0

            ks_stat = None
            if ks_2samp is not None:
                ks_stat = float(ks_2samp(fp_vals, tn_vals).statistic)

            sep_rows.append({
                "feature": feat,
                "feature_column_used": col,
                "median_fp": median_fp,
                "median_tn": median_tn,
                "delta_median": delta_median,
                "smd": float(smd),
                "ks_stat": ks_stat,
            })

        sep_df = pd.DataFrame(sep_rows)
        if not sep_df.empty:
            sep_df["abs_smd"] = sep_df["smd"].abs()
            sep_df = sep_df.sort_values("abs_smd", ascending=False).drop(columns=["abs_smd"])

        sep_path = out_path / f"seed{seed}_{split}_fp_vs_tn_feature_separation.csv"
        sep_df.to_csv(sep_path, index=False)
        artifacts["fp_vs_tn_feature_separation"] = str(sep_path)

        top_path = out_path / f"seed{seed}_{split}_fp_top_drivers.txt"
        with top_path.open("w", encoding="utf-8") as f:
            f.write("Top FP vs TN feature drivers (by |SMD|):\n")
            if sep_df.empty:
                f.write("No features available.\n")
            else:
                top_df = sep_df.head(int(top_k_features))
                for _, row in top_df.iterrows():
                    ks_val = row["ks_stat"]
                    ks_str = f"{ks_val:.4f}" if ks_val is not None else "NA"
                    f.write(
                        f"- {row['feature']} (col={row['feature_column_used']}): "
                        f"SMD={row['smd']:.4f}, KS={ks_str}, delta_median={row['delta_median']:.4f}\n"
                    )
        artifacts["fp_top_drivers"] = str(top_path)

        # Sanity-check metrics
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(
            f"FP diagnosis ({split}) | tp={tp} tn={tn} fp={fp} fn={fn} "
            f"precision={precision:.4f} recall={recall:.4f} specificity={specificity:.4f}"
        )

        return {
            "split": split,
            "threshold": thr,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "artifacts": artifacts,
        }




    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression with selected features"""

        print(f"\n📈 Training Logistic Regression ({self.feature_mode.upper()} features)...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        lr = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            C=1.0,
            n_jobs=-1,
            solver="lbfgs"
        )

        start_time = time.time()

        lr.fit(X_train_scaled, y_train)

        # Calculate training time
        training_time = time.time() - start_time

        y_pred = lr.predict(X_test_scaled)
        y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]

        # Calculate comprehensive metrics
        model_name = f'LR_{self.feature_mode.upper()}'
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba, model_name)

        # Print comprehensive metrics
        self.print_comprehensive_metrics(metrics, f"Logistic Regression ({self.feature_mode.upper()})")
        print(f"   ⏱️  Training time: {training_time:.2f} seconds")

        # Save model and metrics (save scaler along with model)
        model_with_scaler = {'model': lr, 'scaler': self.scaler}
        self.save_model_and_metrics(model_with_scaler, model_name, metrics, training_time, hyperparameters={
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs"
        })

        # Store model and results
        self.models[model_name] = lr
        self.results[model_name] = metrics

        return lr

    def train_neural_network(self, X_train, X_val, X_test, y_train, y_val, y_test,
                             tune_threshold=True,
                             min_recall=0.85,
                             calibrate_probabilities=False, calibration_method="sigmoid",
                             nn_imbalance_mode="baseline",
                             focal_gamma=2.0,
                             downsample_training=False,
                             random_state=42,
                             base_dir=BASE_OUTPUT_DIR):
        """Train Neural Network with selected features and phase-2 threshold tuning."""

        print(f"\nTraining Neural Network ({self.feature_mode.upper()} features)...")

        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        # Check GPU availability for NN training
        if gpu_available:
            print("   Attempting GPU acceleration for Neural Network training")
        else:
            print("   Using optimized CPU for Neural Network training")

        # Optional train-only downsampling
        downsample_info = {"downsampled": False}
        if downsample_training:
            X_fit, y_fit, downsample_info = self._downsample_train_only(
                X_train, y_train, target_pos_frac=0.50, random_state=random_state
            )
        else:
            X_fit, y_fit = X_train, y_train

        # Scale features (fit on train only)
        X_train_scaled = self.scaler.fit_transform(X_fit)
        X_test_scaled = self.scaler.transform(X_test)
        X_val_scaled = self.scaler.transform(X_val)

        # Ensure TF-friendly dtypes to reduce device transfer issues
        X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
        X_val_scaled = np.asarray(X_val_scaled, dtype=np.float32)
        X_test_scaled = np.asarray(X_test_scaled, dtype=np.float32)

        y_fit_arr = y_fit.values if hasattr(y_fit, "values") else np.asarray(y_fit)
        n_pos = int((y_fit_arr == 1).sum())
        n_neg = int((y_fit_arr == 0).sum())

        allowed_modes = {"baseline", "class_weight", "focal", "balanced_batches", "balanced_focal"}
        if nn_imbalance_mode not in allowed_modes:
            raise ValueError(f"Unknown nn_imbalance_mode: {nn_imbalance_mode}")

        class_weight = None
        focal_alpha = None
        loss_fn = "binary_crossentropy"
        batching_strategy = "natural"

        if nn_imbalance_mode == "class_weight":
            w1 = (n_neg / max(n_pos, 1))
            class_weight = {0: 1.0, 1: float(w1)}
        elif nn_imbalance_mode in {"focal", "balanced_focal"}:
            total = max(n_pos + n_neg, 1)
            focal_alpha = float(n_neg / total)

            def focal_loss(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                eps = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
                p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
                alpha_factor = y_true * focal_alpha + (1.0 - y_true) * (1.0 - focal_alpha)
                modulating_factor = tf.pow(1.0 - p_t, focal_gamma)
                return tf.reduce_mean(-alpha_factor * modulating_factor * tf.math.log(p_t))

            loss_fn = focal_loss

        if nn_imbalance_mode in {"balanced_batches", "balanced_focal"}:
            batching_strategy = "balanced_batches"

        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001)
        epoches = 100
        batch_size = 2048

        # Neural network architecture
        def build_model():
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X_train.shape[1],)),

                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),

                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Dense(1, activation="sigmoid")
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=loss_fn,
                metrics=["accuracy"]
            )
            return model

        def run_training(model):
            if batching_strategy == "balanced_batches" and n_pos > 0 and n_neg > 0:
                pos_mask = y_fit_arr == 1
                neg_mask = y_fit_arr == 0
                X_pos = X_train_scaled[pos_mask]
                y_pos = y_fit_arr[pos_mask]
                X_neg = X_train_scaled[neg_mask]
                y_neg = y_fit_arr[neg_mask]

                ds_pos = tf.data.Dataset.from_tensor_slices((X_pos, y_pos)).repeat()
                ds_neg = tf.data.Dataset.from_tensor_slices((X_neg, y_neg)).repeat()
                train_ds = tf.data.Dataset.sample_from_datasets(
                    [ds_pos, ds_neg], weights=[0.5, 0.5], seed=random_state
                )
                train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

                steps_per_epoch = int(np.ceil(len(y_fit_arr) / batch_size))
                history = model.fit(
                    train_ds,
                    epochs=epoches,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=(X_val_scaled, y_val),
                    verbose=0,
                    callbacks=[early_stopping, reduce_lr, EpochLogger(log_every=5)]
                )
            else:
                history = model.fit(
                    X_train_scaled, y_fit_arr,
                    epochs=epoches,
                    batch_size=batch_size,
                    validation_data=(X_val_scaled, y_val),
                    class_weight=class_weight,
                    verbose=0,
                    callbacks=[early_stopping, reduce_lr, EpochLogger(log_every=5)]
                )
            return history

        start_time = time.time()
        training_device = "gpu" if gpu_available else "cpu"
        try:
            model = build_model()
            history = run_training(model)
        except tf.errors.InternalError as e:
            msg = str(e)
            if "Failed copying input tensor" in msg or "Dst tensor is not initialized" in msg:
                print("   GPU training failed; retrying on CPU")
                training_device = "cpu"
                try:
                    tf.config.set_visible_devices([], "GPU")
                except Exception:
                    pass
                tf.keras.backend.clear_session()
                with tf.device("/CPU:0"):
                    model = build_model()
                    history = run_training(model)
            else:
                raise

        # Calculate training time
        training_time = time.time() - start_time

        # Probabilities on VAL and TEST
        val_proba_raw = model.predict(X_val_scaled, verbose=0).flatten()
        test_proba_raw = model.predict(X_test_scaled, verbose=0).flatten()

        # Calibration using VAL
        calibration_meta = {"calibration_applied": False}
        if calibrate_probabilities:
            calibrate_fn, calibration_meta = self._fit_prob_calibrator(
                y_val, val_proba_raw, method=calibration_method
            )
            val_proba = calibrate_fn(val_proba_raw)
            test_proba = calibrate_fn(test_proba_raw)
            calibration_meta["calibration_applied"] = True
        else:
            val_proba = val_proba_raw
            test_proba = test_proba_raw

        # Threshold tuning on VAL
        threshold = 0.5
        threshold_info = {}
        if tune_threshold:
            threshold, threshold_info = self.tune_threshold_max_specificity(
                y_val, val_proba, min_recall=min_recall
            )
            print(f"   Tuned threshold={threshold:.4f} (constraint recall>={min_recall})")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir, tables_dir = self._get_artifact_dirs(base_dir)
        filename_prefix = (
            f"NN_{self.feature_mode.upper()}_{nn_imbalance_mode}_{timestamp}"
            f"_seed{random_state}_minrecall{min_recall}"
        )
        artifacts = {}

        # artifacts["val_curves"] = self._plot_roc_pr_curves(
        #     y_val, val_proba,
        #     out_dir=plots_dir,
        #     title_prefix=f"NN ({self.feature_mode.upper()}) - {nn_imbalance_mode} - VAL",
        #     filename_prefix=f"{filename_prefix}_val",
        #     min_recall=min_recall,
        #     seed=random_state
        # )
        # artifacts["test_curves"] = self._plot_roc_pr_curves(
        #     y_test, test_proba,
        #     out_dir=plots_dir,
        #     title_prefix=f"NN ({self.feature_mode.upper()}) - {nn_imbalance_mode} - TEST",
        #     filename_prefix=f"{filename_prefix}_test",
        #     min_recall=min_recall,
        #     seed=random_state
        # )

        sweep_all_df, sweep_all_meta = self._threshold_sweep_metrics(
            y_val, val_proba,
            thresholds=None,
            min_recall=None
        )
        sweep_all_path = Path(tables_dir) / f"{filename_prefix}_val_threshold_sweep_all.csv"
        #sweep_all_df.to_csv(sweep_all_path, index=False)
        artifacts["val_threshold_sweep_all_csv"] = str(sweep_all_path)
        artifacts["val_threshold_sweep_all_meta"] = sweep_all_meta

        sweep_con_df, sweep_con_meta = self._threshold_sweep_metrics(
            y_val, val_proba,
            thresholds=None,
            min_recall=min_recall if tune_threshold else None
        )
        sweep_con_path = Path(tables_dir) / f"{filename_prefix}_val_threshold_sweep_constrained.csv"
        sweep_con_df.to_csv(sweep_con_path, index=False)
        artifacts["val_threshold_sweep_constrained_csv"] = str(sweep_con_path)
        artifacts["val_threshold_sweep_constrained_meta"] = sweep_con_meta

        # Apply to TEST
        y_pred = (test_proba >= threshold).astype(int)

        model_name = f"NN_{self.feature_mode.upper()}_{nn_imbalance_mode.upper()}"
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, test_proba, model_name)

        # Print comprehensive metrics
        self.print_comprehensive_metrics(
            metrics, f"Neural Network ({self.feature_mode.upper()}) - {nn_imbalance_mode}"
        )
        print(f"   Training time: {training_time:.2f} seconds")

        nn_hyperparameters = {
            "model_type": "NN",
            "input_dim": X_train.shape[1],
            "layers": [
                ["Dense 128 relu", "BatchNormalization", "Dropout 0.4"],
                ["Dense 64 relu", "BatchNormalization", "Dropout 0.3"],
                ["Dense 1 sigmoid"]
            ],
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "batch_size": batch_size,
            "epochs": epoches,
            "loss": "focal_loss" if nn_imbalance_mode in {"focal", "balanced_focal"} else "binary_crossentropy",
            "nn_imbalance_mode": nn_imbalance_mode,
            "class_weight": class_weight,
            "focal_gamma": float(focal_gamma) if focal_alpha is not None else None,
            "focal_alpha": float(focal_alpha) if focal_alpha is not None else None,
            "batching_strategy": batching_strategy,
            "early_stopping": 10,
            "reduce_lr_factor": 0.2,
            "reduce_lr_patience": 5,
            "training_device": training_device,
            "threshold_tuning": bool(tune_threshold),
            "decision_threshold": float(threshold),
            "probability_calibration": dict(calibration_meta),
            "train_downsample": dict(downsample_info),
            **threshold_info
        }

        # Evaluate with a safe CPU fallback if GPU transfer fails late in the run
        try:
            test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        except tf.errors.InternalError as e:
            msg = str(e)
            if "Failed copying input tensor" in msg or "Dst tensor is not initialized" in msg:
                print("   GPU evaluation failed; retrying on CPU")
                with tf.device("/CPU:0"):
                    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            else:
                raise

        self.save_model_and_metrics(
            model,
            model_name,
            metrics,
            training_time,
            base_dir=base_dir,
            hyperparameters=nn_hyperparameters,
            training_history=history,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            extra_artifacts=artifacts,
            timestamp=timestamp
        )

        self.models[model_name] = model
        self.results[model_name] = metrics
        self.model = model
        return model

    def train_cnn(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train Convolutional Neural Network classifier"""

        print("\n🧠 Training Convolutional Neural Network...")

        # Check GPU availability for CNN training
        if gpu_available:
            print("   🎮 Attempting GPU acceleration for CNN training")
        else:
            print("   💻 Using optimized CPU for CNN training")

        model_name = f'CNN_{self.feature_mode.upper()}'

        # Scale features for CNN
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Reshape for CNN (add channel dimension)
        X_train_cnn = X_train_scaled[..., np.newaxis]
        X_val_cnn = X_val_scaled[..., np.newaxis]
        X_test_cnn = X_test_scaled[..., np.newaxis]

        # Create CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1], 1)),

            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=3,
                padding="same",
                activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                padding="same",
                activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),

            # 🔑 SAFE pooling for any feature length
            tf.keras.layers.GlobalMaxPooling1D(),

            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        epoches = 100
        batch_size = 2048

        start_time = time.time()
        # Train model
        history = model.fit(
            X_train_cnn, y_train,
            epochs=epoches,
            batch_size=batch_size,
            validation_data=(X_val_cnn, y_val),
            callbacks=[early_stopping, reduce_lr, EpochLogger(log_every=5)],
            verbose=0
        )
        # Calculate training time
        training_time = time.time() - start_time

        # Predictions
        y_pred_proba = model.predict(X_test_cnn).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba, model_name)

        # Print comprehensive metrics
        self.print_comprehensive_metrics(metrics, f"CNN ({self.feature_mode.upper()})")
        print(f"   ⏱️  Training time: {training_time:.2f} seconds")

        cnn_hyperparameters = {
            "model_type": "CNN",
            "input_dim": X_train.shape[1],
            "conv_layers": [
                {"filters": 32, "kernel_size": 3, "activation": "relu"},
                {"filters": 64, "kernel_size": 3, "activation": "relu"}
            ],
            "dense_layers": [64, 1],
            "dropout": 0.3,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "loss":'binary_crossentropy',
            "batch_size": batch_size,
            "epochs": epoches,
            "early_stopping": 10,
            "reduce_lr": 0.2
        }

        test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)

        # Save model and metrics
        self.save_model_and_metrics(model, model_name, metrics, training_time, hyperparameters=cnn_hyperparameters,
                                    training_history=history, test_loss=test_loss, test_accuracy=test_accuracy)

        # Store model and results
        self.models[model_name] = model
        self.results[model_name] = metrics

        return model


    def train_tabtransformer(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train a Transformer-style model for numeric tabular data"""

        print(f"\n🤖 Training TabTransformer ({self.feature_mode.upper()} features)...")
        model_name = f"TabTransformer_{self.feature_mode.upper()}"

        # Scale numeric features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_val_scaled = self.scaler.transform(X_val)

        num_features = X_train_scaled.shape[1]
        hidden_dim = 64
        num_heads = 4
        num_blocks = 2

        inputs = tf.keras.Input(shape=(num_features,), name="features")

        # Project features into hidden space
        x = tf.keras.layers.Dense(hidden_dim, activation="relu")(inputs)
        x = tf.keras.layers.LayerNormalization()(x)

        # Transformer-style feature interaction blocks
        for _ in range(num_blocks):
            attn_input = tf.expand_dims(x, axis=1)  # (batch, 1, hidden_dim)

            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=hidden_dim // num_heads,
                dropout=0.2
            )(attn_input, attn_input)

            attn_output = tf.squeeze(attn_output, axis=1)

            x = tf.keras.layers.Add()([x, attn_output])
            x = tf.keras.layers.LayerNormalization()(x)

            x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)

        # Classification head
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs, outputs, name="TabTransformer")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        batch_size = 2048
        epoches = 100

        start_time = time.time()

        history = model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epoches,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stopping, reduce_lr, EpochLogger(log_every=5)]
        )

        training_time = time.time() - start_time

        tabtransformer_hyperparameters = {
                "model_type": "TabTransformer",
                "input_dim": num_features,
                "dense_dim": hidden_dim,
                "num_attention_blocks": num_blocks,
                "num_heads": num_heads,
                "key_dim": hidden_dim // num_heads,
                "dropout": 0.2,
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "batch_size": batch_size,
                "epochs": epoches,
                "loss":'binary_crossentropy',
                "early_stopping": 10,
                "reduced_lr": 0.2
            }

        y_pred_proba = model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = self.calculate_comprehensive_metrics(
            y_test, y_pred, y_pred_proba, model_name
        )
        self.print_comprehensive_metrics(metrics, f"TabTransformer ({self.feature_mode.upper()})")

        test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

        self.save_model_and_metrics(
            model,
            model_name,
            metrics,
            training_time,
            feature_names=self.feature_names,
            feature_importance=None,  # not applicable,
            hyperparameters=tabtransformer_hyperparameters,
            training_history=history,
            test_loss=test_loss,
            test_accuracy=test_accuracy
        )

        self.models[model_name] = model
        self.results[model_name] = metrics

        return model

    def _get_artifact_dirs(self, base_dir: str):
        """
        Create (if needed) and return standard output directories for phase-2 artifacts.
        Returns a dict with keys: plots_dir, tables_dir.
        """
        ml_results_dir = os.path.join(base_dir, "ml_results")
        plots_dir = os.path.join(ml_results_dir, "plots")
        tables_dir = os.path.join(ml_results_dir, "tables")

        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        return plots_dir, tables_dir

    def _plot_roc_pr_curves(
            self,
            y_true,
            y_prob,
            out_dir: str,
            title_prefix: str,
            filename_prefix: str,
            min_recall,
            seed
    ):
        """
        Plot ROC and PR curves and save PNGs.

        Args:
            y_true: array-like of shape (n_samples,), binary labels {0,1}
            y_prob: array-like of shape (n_samples,), predicted probabilities for class 1
            out_dir: directory to save the PNG files
            title_prefix: prefix used in plot titles
            filename_prefix: prefix used in filenames (without extension)

        Returns:
            {
              "roc_path": "...png",
              "pr_path":  "...png",
              "roc_auc": float,
              "pr_auc":  float  # average precision
            }
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            roc_curve,
            auc,
            precision_recall_curve,
            average_precision_score,
        )

        os.makedirs(out_dir, exist_ok=True)

        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)

        # --- ROC ---
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = float(auc(fpr, tpr))
        roc_path = os.path.join(out_dir, f"{filename_prefix}_roc_{min_recall}_seed{seed}.png")

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title_prefix} - ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=200)
        plt.close()

        # --- PR ---
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(average_precision_score(y_true, y_prob))
        pr_path = os.path.join(out_dir, f"{filename_prefix}_pr_{min_recall}_seed{seed}.png")

        plt.figure()
        plt.plot(recall, precision, label=f"AP={pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{title_prefix} - PR")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(pr_path, dpi=200)
        plt.close()

        return {
            "roc_path": roc_path,
            "pr_path": pr_path,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

    def _threshold_sweep_metrics(
            self,
            y_true,
            y_prob,
            thresholds=None,
            min_recall: float = None
    ):
        """
        Compute metrics for many thresholds.

        Returns:
            (df, meta)

        df columns:
          threshold, tp, fp, tn, fn, accuracy, precision, recall, specificity, f1, balanced_accuracy
          plus meets_min_recall if min_recall is not None

        meta includes:
          n_thresholds, min_threshold, max_threshold, min_recall, n_meeting_min_recall,
          best_threshold_by_specificity (within constraint if provided), and its recall/specificity.
        """
        import numpy as np
        import pandas as pd

        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)

        if thresholds is None:
            # Quantile-based grid keeps the sweep bounded even for huge datasets.
            qs = np.linspace(0.0, 1.0, 401)  # step=0.0025
            thresholds = np.unique(np.quantile(y_prob, qs))
            thresholds = np.unique(np.concatenate([thresholds, [0.0, 1.0]]))

        rows = []
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)

            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())

            denom = (tp + tn + fp + fn)
            acc = (tp + tn) / denom if denom else 0.0

            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0

            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
            bal_acc = 0.5 * (rec + spec)

            rows.append(
                {
                    "threshold": float(thr),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "specificity": spec,
                    "f1": f1,
                    "balanced_accuracy": bal_acc,
                }
            )

        df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

        meta = {
            "n_thresholds": int(len(df)),
            "min_threshold": float(df["threshold"].min()) if len(df) else None,
            "max_threshold": float(df["threshold"].max()) if len(df) else None,
            "min_recall": float(min_recall) if min_recall is not None else None,
        }

        if min_recall is not None:
            df["meets_min_recall"] = df["recall"] >= float(min_recall)
            n_meet = int(df["meets_min_recall"].sum())
            meta["n_meeting_min_recall"] = n_meet

            if n_meet > 0:
                # Best threshold = maximize specificity under recall constraint
                sub = df[df["meets_min_recall"]].copy()
                best_row = sub.loc[sub["specificity"].idxmax()]
                meta.update({
                    "best_threshold_by_specificity": float(best_row["threshold"]),
                    "best_specificity": float(best_row["specificity"]),
                    "best_recall": float(best_row["recall"]),
                })
            else:
                meta.update({
                    "best_threshold_by_specificity": None,
                    "best_specificity": None,
                    "best_recall": None,
                })
        else:
            # Unconstrained best-by-specificity (mostly for inspection)
            if len(df) > 0:
                best_row = df.loc[df["specificity"].idxmax()]
                meta.update({
                    "best_threshold_by_specificity": float(best_row["threshold"]),
                    "best_specificity": float(best_row["specificity"]),
                    "best_recall": float(best_row["recall"]),
                })

        return df, meta

    def _compute_specificity(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def tune_threshold_max_specificity(self, y_true, y_proba, min_recall=MIN_RECALL):
        """
        Pick threshold that maximizes specificity subject to recall >= min_recall.
        Returns: best_threshold, dict(val_constraint_metrics)
        """
        # Candidate thresholds: use unique probs for exact evaluation (can downsample if huge)
        thresholds = np.unique(y_proba)
        # To speed up on very large val sets, sample thresholds:
        if len(thresholds) > 2000:
            # evenly spaced percentiles
            qs = np.linspace(0, 1, 2000)
            thresholds = np.quantile(y_proba, qs)

        best = {
            "threshold": 0.5,
            "specificity": -1.0,
            "recall": 0.0
        }

        # Vectorized-ish loop (still OK because thresholds <= 2000)
        y_true = np.asarray(y_true)

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)

            # recall = TP/(TP+FN)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if recall < min_recall:
                continue

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            if specificity > best["specificity"]:
                best["threshold"] = float(t)
                best["specificity"] = float(specificity)
                best["recall"] = float(recall)

        return best["threshold"], {
            "val_threshold_constraint": f"maximize_specificity_given_recall>={min_recall}",
            "val_best_threshold": best["threshold"],
            "val_specificity_at_threshold": best["specificity"],
            "val_recall_at_threshold": best["recall"],
            "min_recall_constraint": float(min_recall),
        }

    def _downsample_train_only(self, X_train, y_train, target_pos_frac=0.50, random_state=RANDOM_STATE):
        """
        Train-only random downsampling of the MAJORITY class to achieve approx target_pos_frac.
        Works whether positives or negatives are majority.

        target_pos_frac=0.50 -> aim for ~50% positives in training data
        """
        if target_pos_frac is None:
            return X_train, y_train, {"downsampled": False}

        y = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        n_pos = len(pos_idx)
        n_neg = len(neg_idx)

        if n_pos == 0 or n_neg == 0:
            return X_train, y_train, {"downsampled": False, "reason": "single_class_train"}

        rng = np.random.default_rng(random_state)

        # Desired relationship:
        # pos / (pos + neg) = target_pos_frac
        # Solve for neg given pos: neg = pos*(1-target)/target
        # Solve for pos given neg: pos = neg*target/(1-target)
        target = float(target_pos_frac)

        if n_pos / (n_pos + n_neg) > target:
            # Too many positives -> downsample positives
            n_pos_target = int(n_neg * target / max(1e-9, (1 - target)))
            n_pos_sample = max(1, min(n_pos, n_pos_target))

            pos_sample_idx = rng.choice(pos_idx, size=n_pos_sample, replace=False)
            keep_idx = np.concatenate([pos_sample_idx, neg_idx])
            action = "downsample_pos"
        else:
            # Too many negatives -> downsample negatives
            n_neg_target = int(n_pos * (1 - target) / max(target, 1e-9))
            n_neg_sample = max(1, min(n_neg, n_neg_target))

            neg_sample_idx = rng.choice(neg_idx, size=n_neg_sample, replace=False)
            keep_idx = np.concatenate([pos_idx, neg_sample_idx])
            action = "downsample_neg"

        rng.shuffle(keep_idx)

        # Preserve pandas indexing
        if hasattr(X_train, "iloc"):
            X_ds = X_train.iloc[keep_idx]
            y_ds = y_train.iloc[keep_idx]
        else:
            X_ds = X_train[keep_idx]
            y_ds = y_train[keep_idx]

        y_ds_arr = y_ds.values if hasattr(y_ds, "values") else np.asarray(y_ds)
        after_n_pos = int((y_ds_arr == 1).sum())
        after_n_neg = int((y_ds_arr == 0).sum())

        info = {
            "downsampled": True,
            "action": action,
            "target_pos_frac": target,
            "before_n_pos": int(n_pos),
            "before_n_neg": int(n_neg),
            "after_n_pos": after_n_pos,
            "after_n_neg": after_n_neg,
            "after_pos_frac": float(after_n_pos / (after_n_pos + after_n_neg))
        }
        return X_ds, y_ds, info

    def _fit_prob_calibrator(self, y_cal, proba_cal, method="sigmoid"):
        """
        Fit a probability calibration model on validation set probabilities.
        Returns a callable f(p)->p_cal and a dict of calibration metadata.
        """
        y = y_cal.values if hasattr(y_cal, "values") else np.asarray(y_cal)
        p = np.asarray(proba_cal).astype(float)

        # Guard: if only one class in validation, calibration is undefined
        if len(np.unique(y)) < 2:
            return (lambda x: np.asarray(x)), {"calibration": "skipped_one_class_val"}

        if method == "sigmoid":
            # Platt scaling: logistic regression on 1D probability feature
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(p.reshape(-1, 1), y)

            def calibrate_fn(x):
                x = np.asarray(x).astype(float).reshape(-1, 1)
                return lr.predict_proba(x)[:, 1]

            meta = {
                "calibration_method": "sigmoid",
                "calibration_model": "LogisticRegression(PlattScaling)",
                "coef": float(lr.coef_.ravel()[0]),
                "intercept": float(lr.intercept_.ravel()[0]),
            }
            return calibrate_fn, meta

        elif method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)

            def calibrate_fn(x):
                x = np.asarray(x).astype(float)
                return iso.transform(x)

            meta = {"calibration_method": "isotonic", "calibration_model": "IsotonicRegression"}
            return calibrate_fn, meta

        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def train_lightgbm(self, X_train, X_val, X_test, y_train, y_val, y_test,
                       use_scale_pos_weight=True,
                       tune_threshold=True,
                       min_recall=MIN_RECALL,
                       # NEW:
                       calibrate_probabilities=True,
                       calibration_method="sigmoid",  # "sigmoid" or "isotonic"
                       train_downsample=False,
                       downsample_target_pos_frac=0.50,
                       downsample_random_state=RANDOM_STATE,
                       # NEW: where to save plots/tables (keeps existing default structure)
                       base_dir=BASE_OUTPUT_DIR):
        """Train LightGBM with optional scale_pos_weight + threshold tuning + (NEW) calibration + (NEW) train-only downsampling."""

        print(f"\n💡 Training LightGBM ({self.feature_mode.upper()} features).")
        model_name = f"LGBM_{self.feature_mode.upper()}"

        # Sanitize feature names (keep your existing behavior)
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()
        X_train.columns = X_train.columns.str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)
        X_val.columns = X_train.columns
        X_test.columns = X_train.columns

        # ---- (NEW) train-only downsampling ----
        downsample_info = {"downsampled": False}
        if train_downsample:
            X_fit, y_fit, downsample_info = self._downsample_train_only(
                X_train, y_train,
                target_pos_frac=downsample_target_pos_frac,
                random_state=downsample_random_state
            )
        else:
            X_fit, y_fit = X_train, y_train

        # ---- scale_pos_weight (computed on the ACTUAL training data used to fit) ----
        n_pos = int((y_fit == 1).sum())
        n_neg = int((y_fit == 0).sum())
        spw = (n_neg / max(n_pos, 1)) if use_scale_pos_weight else 1.0

        lgbm = lgb.LGBMClassifier(
            n_estimators=1000,  # same as phase-1 upper bound
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=50,
            colsample_bytree=0.8,
            subsample=0.8,
            early_stopping_rounds=50,  # same as phase-1
            objective='binary',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            scale_pos_weight=float(spw) if use_scale_pos_weight else None
        )

        start_time = time.time()
        lgbm.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
        )
        training_time = time.time() - start_time

        # ---- raw probabilities ----
        val_proba_raw = lgbm.predict_proba(X_val)[:, 1]
        test_proba_raw = lgbm.predict_proba(X_test)[:, 1]

        # ---- (NEW) calibration using VAL ----
        calibration_meta = {"calibration_applied": False}
        if calibrate_probabilities:
            calibrate_fn, calibration_meta = self._fit_prob_calibrator(
                y_val, val_proba_raw, method=calibration_method
            )
            val_proba = calibrate_fn(val_proba_raw)
            test_proba = calibrate_fn(test_proba_raw)
            calibration_meta["calibration_applied"] = True
        else:
            val_proba = val_proba_raw
            test_proba = test_proba_raw

        # ---- threshold tuning on VAL (use CALIBRATED probabilities if enabled) ----
        threshold = 0.5
        threshold_info = {}
        if tune_threshold:
            threshold, threshold_info = self.tune_threshold_max_specificity(
                y_val, val_proba, min_recall=min_recall
            )
            print(f"   🎯 Tuned threshold={threshold:.4f} (constraint recall>={min_recall})")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # ---- (NEW) Plot ROC/PR curves + record metrics across thresholds ----
        plots_dir, tables_dir = self._get_artifact_dirs(base_dir)
        filename_prefix = f"{model_name}_{self.feature_mode}_{timestamp}"
        artifacts = {}

        # # Curves on VAL (the split used for threshold tuning)
        # artifacts["val_curves"] = self._plot_roc_pr_curves(
        #     y_val, val_proba,
        #     out_dir=plots_dir,
        #     title_prefix=f"{model_name} ({self.feature_mode}) - VAL",
        #     filename_prefix=f"{filename_prefix}_val",
        #     min_recall=min_recall,
        #     seed=RANDOM_STATE
        # )

        # # Curves on TEST (final evaluation)
        # artifacts["test_curves"] = self._plot_roc_pr_curves(
        #     y_test, test_proba,
        #     out_dir=plots_dir,
        #     title_prefix=f"{model_name} ({self.feature_mode}) - TEST",
        #     filename_prefix=f"{filename_prefix}_test",
        #     min_recall=min_recall,
        #     seed=RANDOM_STATE
        # )

        # Threshold sweep tables (VAL): full + constrained (min_recall)
        sweep_all_df, sweep_all_meta = self._threshold_sweep_metrics(
            y_val, val_proba,
            thresholds=None,
            min_recall=None
        )
        sweep_all_path = Path(tables_dir) / f"{filename_prefix}_val_threshold_sweep_all.csv"
        # sweep_all_df.to_csv(sweep_all_path, index=False)
        artifacts["val_threshold_sweep_all_csv"] = str(sweep_all_path)
        artifacts["val_threshold_sweep_all_meta"] = sweep_all_meta

        sweep_con_df, sweep_con_meta = self._threshold_sweep_metrics(
            y_val, val_proba,
            thresholds=None,
            min_recall=min_recall if tune_threshold else None
        )
        sweep_con_path = Path(tables_dir) / f"{filename_prefix}_val_threshold_sweep_constrained_{min_recall}_seed{RANDOM_STATE}.csv"
        sweep_con_df.to_csv(sweep_con_path, index=False)
        artifacts["val_threshold_sweep_constrained_csv"] = str(sweep_con_path)
        artifacts["val_threshold_sweep_constrained_meta"] = sweep_con_meta

        # Apply to TEST
        y_pred = (test_proba >= threshold).astype(int)
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, test_proba, model_name)

        self.print_comprehensive_metrics(metrics, f"LightGBM ({self.feature_mode.upper()})")
        print(f"   🌳 Best iteration: {getattr(lgbm, 'best_iteration_', None)}")
        print(f"   ⏱️ Training time: {training_time:.2f}s")

        lgbm_hparams = {
            **lgbm.get_params(),
            "best_iteration": getattr(lgbm, "best_iteration_", None),
            "use_scale_pos_weight": bool(use_scale_pos_weight),
            "computed_scale_pos_weight": float(spw),
            "threshold_tuning": bool(tune_threshold),
            "decision_threshold": float(threshold),
            "probability_calibration": dict(calibration_meta),
            "train_downsample": dict(downsample_info),
            **threshold_info
        }

        self.save_model_and_metrics(
            lgbm,
            model_name,
            metrics,
            training_time,
            base_dir=base_dir,
            hyperparameters=lgbm_hparams,
            feature_names=self.feature_names,
            feature_importance=lgbm.feature_importances_,
            extra_artifacts=artifacts,
            timestamp=timestamp
        )

        self.models[model_name] = lgbm
        self.results[model_name] = metrics
        self.model = lgbm
        self.decision_threshold = float(threshold)
        return lgbm

    def train_random_forest(self, X_train, X_val, X_test, y_train, y_val, y_test,
                            class_weight="balanced",
                            tune_threshold=True,
                            min_recall=MIN_RECALL,
                            # NEW:
                            calibrate_probabilities=True,
                            calibration_method="sigmoid",
                            train_downsample=False,
                            downsample_target_pos_frac=0.50,
                            downsample_random_state=RANDOM_STATE,
                            # NEW: where to save plots/tables/models
                            base_dir=BASE_OUTPUT_DIR):
        """Train Random Forest with early stopping + threshold tuning + (NEW) calibration + (NEW) train-only downsampling."""

        print(f"\n🌳 Training Random Forest ({self.feature_mode.upper()} features).")
        from copy import deepcopy
        from sklearn.metrics import roc_auc_score

        model_name = f'RF_{self.feature_mode.upper()}'

        # ---- (NEW) train-only downsampling ----
        downsample_info = {"downsampled": False}
        if train_downsample:
            X_fit, y_fit, downsample_info = self._downsample_train_only(
                X_train, y_train,
                target_pos_frac=downsample_target_pos_frac,
                random_state=downsample_random_state
            )
        else:
            X_fit, y_fit = X_train, y_train

        max_estimators = 600
        step = 50
        patience = 2
        best_val_auc = -np.inf
        patience_counter = 0

        rf = RandomForestClassifier(
            n_estimators=0,
            max_depth=12,
            min_samples_split=100,
            min_samples_leaf=50,
            max_features='sqrt',
            warm_start=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=class_weight
        )

        start_time = time.time()
        best_rf = None
        best_n_estimators = 0

        for n_estimators in range(step, max_estimators + 1, step):
            rf.set_params(n_estimators=n_estimators)
            rf.fit(X_fit, y_fit)

            val_pred_proba = rf.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred_proba)

            print(f"   🌱 Trees: {n_estimators:3d} | Val ROC-AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_rf = deepcopy(rf)
                best_n_estimators = n_estimators
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"   🛑 Early stopping triggered at {n_estimators} trees")
                break

        training_time = time.time() - start_time
        rf = best_rf

        # ---- raw probabilities from BEST model ----
        val_proba_raw = rf.predict_proba(X_val)[:, 1]
        test_proba_raw = rf.predict_proba(X_test)[:, 1]

        # ---- (NEW) calibration using VAL ----
        calibration_meta = {"calibration_applied": False}
        if calibrate_probabilities:
            calibrate_fn, calibration_meta = self._fit_prob_calibrator(
                y_val, val_proba_raw, method=calibration_method
            )
            val_proba = calibrate_fn(val_proba_raw)
            test_proba = calibrate_fn(test_proba_raw)
            calibration_meta["calibration_applied"] = True
        else:
            val_proba = val_proba_raw
            test_proba = test_proba_raw

        # ---- threshold tuning on VAL (use CALIBRATED probabilities if enabled) ----
        threshold = 0.5
        threshold_info = {}
        if tune_threshold:
            threshold, threshold_info = self.tune_threshold_max_specificity(
                y_val, val_proba, min_recall=min_recall
            )
            print(f"   🎯 Tuned threshold={threshold:.4f} (constraint recall>={min_recall})")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ---- (NEW) Plot ROC/PR curves + record metrics across thresholds ----
        plots_dir, tables_dir = self._get_artifact_dirs(base_dir)
        filename_prefix = f"{model_name}_{self.feature_mode}_{timestamp}"
        artifacts = {}

        # artifacts["val_curves"] = self._plot_roc_pr_curves(
        #     y_val, val_proba,
        #     out_dir=plots_dir,
        #     title_prefix=f"{model_name} ({self.feature_mode}) - VAL",
        #     filename_prefix=f"{filename_prefix}_val"
        # )
        # artifacts["test_curves"] = self._plot_roc_pr_curves(
        #     y_test, test_proba,
        #     out_dir=plots_dir,
        #     title_prefix=f"{model_name} ({self.feature_mode}) - TEST",
        #     filename_prefix=f"{filename_prefix}_test"
        # )

        sweep_all_df, sweep_all_meta = self._threshold_sweep_metrics(
            y_val, val_proba,
            thresholds=None,
            min_recall=None
        )
        sweep_all_path = Path(tables_dir) / f"{filename_prefix}_val_threshold_sweep_all.csv"
        #sweep_all_df.to_csv(sweep_all_path, index=False)
        artifacts["val_threshold_sweep_all_csv"] = str(sweep_all_path)
        artifacts["val_threshold_sweep_all_meta"] = sweep_all_meta

        sweep_con_df, sweep_con_meta = self._threshold_sweep_metrics(
            y_val, val_proba,
            thresholds=None,
            min_recall=min_recall if tune_threshold else None
        )
        sweep_con_path = Path(tables_dir) / f"{filename_prefix}_val_threshold_sweep_constrained.csv"
        sweep_con_df.to_csv(sweep_con_path, index=False)
        artifacts["val_threshold_sweep_constrained_csv"] = str(sweep_con_path)
        artifacts["val_threshold_sweep_constrained_meta"] = sweep_con_meta

        # Apply to TEST
        y_pred = (test_proba >= threshold).astype(int)
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, test_proba, model_name)

        self.print_comprehensive_metrics(metrics, f"Random Forest ({self.feature_mode.upper()})")
        print(f"   🌲 Best trees: {best_n_estimators}")
        print(f"   ⏱️  Training time: {training_time:.2f} seconds")

        rf_hyperparameters = {
            "model_type": "RandomForest",
            "max_depth": 12,
            "min_samples_split": 100,
            "min_samples_leaf": 50,
            "max_features": "sqrt",
            "max_estimators": max_estimators,
            "best_n_estimators": best_n_estimators,
            "early_stopping_patience": patience,
            "validation_metric": "roc_auc",
            "best_validation_roc_auc": float(best_val_auc),
            "class_weight": class_weight,
            "threshold_tuning": bool(tune_threshold),
            "decision_threshold": float(threshold),
            "probability_calibration": dict(calibration_meta),
            "train_downsample": dict(downsample_info),
            **threshold_info
        }

        self.save_model_and_metrics(
            rf,
            model_name,
            metrics,
            training_time,
            base_dir=base_dir,
            feature_names=self.feature_names,
            hyperparameters=rf_hyperparameters,
            feature_importance=rf.feature_importances_,
            extra_artifacts=artifacts
        )

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n📊 Top 15 Important Features ({self.feature_mode.upper()}):")
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']}: {row['importance']:.4f}")

        self.models[model_name] = rf
        self.results[model_name] = metrics
        return rf, feature_importance

    def compare_models(self):
        """Compare all model results with comprehensive metrics"""

        print(f"\n📊 COMPREHENSIVE MODEL COMPARISON ({self.feature_mode.upper()} FEATURES)")
        print("=" * 120)
        print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'MCC':<10}")
        print("-" * 120)

        for model_name, metrics in self.results.items():
            print(f"{model_name:<15} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['roc_auc']:<10.4f} "
                  f"{metrics['pr_auc']:<10.4f} {metrics['mcc']:<10.4f}")

        # Best model by different metrics
        if self.results:
            best_accuracy = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
            best_f1 = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
            best_roc_auc = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
            best_pr_auc = max(self.results.keys(), key=lambda x: self.results[x]['pr_auc'])
            best_mcc = max(self.results.keys(), key=lambda x: self.results[x]['mcc'])

            print(f"\n🏆 BEST MODELS BY METRIC ({self.feature_mode.upper()}):")
            print(f"   🎯 Accuracy:  {best_accuracy} ({self.results[best_accuracy]['accuracy']:.4f})")
            print(f"   🎯 F1-Score:  {best_f1} ({self.results[best_f1]['f1_score']:.4f})")
            print(f"   🎯 ROC-AUC:   {best_roc_auc} ({self.results[best_roc_auc]['roc_auc']:.4f})")
            print(f"   🎯 PR-AUC:    {best_pr_auc} ({self.results[best_pr_auc]['pr_auc']:.4f})")
            print(f"   🎯 MCC:       {best_mcc} ({self.results[best_mcc]['mcc']:.4f})")

            # Overall best (using F1-score as primary metric for imbalanced data)
            best_model = best_f1
            print(f"\n🥇 OVERALL BEST MODEL: {best_model}")
            print(f"   Primary Metric (F1): {self.results[best_model]['f1_score']:.4f}")
            print(f"   Accuracy: {self.results[best_model]['accuracy']:.4f}")
            print(f"   ROC-AUC: {self.results[best_model]['roc_auc']:.4f}")
        else:
            best_model = None
            print(f"\n⚠️  No models trained yet")

        return best_model

    # Usage:
    # classifier.run_enhanced_pipeline(
    #     min_recalls=[0.95, 0.9, 0.85, 0.8],
    #     models=["nn"],
    #     nn_imbalance_modes=["baseline", "class_weight", "focal", "balanced_batches", "balanced_focal"]
    # )
    def run_enhanced_pipeline(self, min_recalls, models=None, nn_imbalance_modes=None,
                              nn_min_recalls=None, nn_random_state=None,
                              save_enhanced_feature_files=SAVE_ENHANCED_FILES):
        """Run the complete enhanced classification pipeline"""

        # Load and merge data
        if models is None:
            models = ["rf", "lr", "nn", "cnn", "lgbm", "tbt"]

        if nn_imbalance_modes is None:
            nn_imbalance_modes = ["baseline"]

        if nn_min_recalls is None:
            nn_min_recalls = min_recalls

        if nn_random_state is None:
            nn_random_state = RANDOM_STATE

        self.load_and_merge_data(
            num_files=NUM_FILES_EACH_FOLDER,
            save_enhanced_feature_files=save_enhanced_feature_files,
        )  # Start with 3 files for testing

        # Prepare features
        poi_features, road_features, intersection_features, time_features = self.prepare_features()

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()

        # ✅ NEW: train-only NAICS priors (no leakage)
        if getattr(self, "enable_naics_features", False):
            X_train, X_val, X_test = self.add_naics_priors_post_split(X_train, X_val, X_test, prefix="naics_")

            # Update stored splits so FP diagnosis uses the same matrices
            self.X_train, self.X_val, self.X_test = X_train, X_val, X_test

            # Quick sanity print
            naics_cols_now = [c for c in X_train.columns if c.startswith("naics_")]
            print(f"✅ NAICS priors added. Total NAICS engineered cols now: {len(naics_cols_now)}")

            print("NAICS cols sample:", [c for c in X_train.columns if c.startswith("naics_")][:10])
            assert "naics_sector_freq_mean" in X_train.columns
            assert "naics_sector_freq_max" in X_train.columns

        feature_importance = None
        rf_min_recall = min_recalls[0] if min_recalls else MIN_RECALL

        # Train models
        if "rf" in models:
            _, feature_importance = self.train_random_forest(
                X_train, X_val, X_test, y_train, y_val, y_test,
                class_weight="balanced", tune_threshold=True, min_recall=rf_min_recall,
                calibration_method="sigmoid",
                train_downsample=False,
                downsample_target_pos_frac=0.50
            )
        if "lr" in models:
            self.train_logistic_regression(X_train, X_test, y_train, y_test)
        if "nn" in models:
            for nn_mode in nn_imbalance_modes:
                for nn_min_recall in nn_min_recalls:
                    self.train_neural_network(
                        X_train, X_val, X_test, y_train, y_val, y_test,
                        tune_threshold=True,
                        min_recall=nn_min_recall,
                        calibrate_probabilities=False,
                        calibration_method="sigmoid",
                        nn_imbalance_mode=nn_mode,
                        focal_gamma=2.0,
                        downsample_training=False,
                        random_state=nn_random_state
                    )
        if "cnn" in models:
            self.train_cnn(X_train, X_val, X_test, y_train, y_val, y_test)

        if "lgbm" in models:
            for min_recall in min_recalls:
                self.train_lightgbm(
                    X_train, X_val, X_test, y_train, y_val, y_test, use_scale_pos_weight=False,
                    tune_threshold=True, min_recall=min_recall, calibrate_probabilities=False,
                    calibration_method="isotonic", train_downsample=False
                )
        if "tbt" in models:
            self.train_tabtransformer(X_train, X_val, X_test, y_train, y_val, y_test)

        # Compare results
        best_model = self.compare_models()

        print(f"PIPELINE COMPLETED ({self.feature_mode.upper()} FEATURES)!")
        print("=" * 70)

        return {
            "feature_mode": self.feature_mode,
            "models": self.models,
            "results": self.results,
            "feature_importance": feature_importance,
            "best_model": best_model,
            "poi_features": poi_features,
            "road_features": road_features,
            "intersection_features": intersection_features,
            "time_features": time_features,
            "selected_features": self.feature_names
        }

def main():
    """Main function - demonstrates all three feature modes"""

    print("🚀 ENHANCED WORK STOP CLASSIFICATION")
    print("=" * 70)

    display_system_info()

    # Data directories
    poi_data_dir = BASE_INPUT_DIR / "stop_records_with_poi_features"
    road_data_dir = BASE_INPUT_DIR / "stop_records_with_road_features"
    intersection_data_dir = BASE_INPUT_DIR / "stop_records_with_intersection_features"
    time_data_dir = BASE_INPUT_DIR / "stop_records_with_time_features"

    # Test all three feature modes
    #feature_modes = ['poi_only', 'road_only', 'both']
    feature_modes = ['all']
    all_results = {}
    intersection_feature_set_mode = 'core'

    for mode in feature_modes:
        print(f"\n{'='*80}")
        print(f"🎯 TESTING FEATURE MODE: {mode.upper()}")
        print(f"{'='*80}")

        try:
            # Create classifier for this mode
            if mode == 'poi_only':
                classifier = EnhancedWorkStopClassifier(poi_data_dir=poi_data_dir, feature_mode=mode)
            elif mode == 'road_only':
                classifier = EnhancedWorkStopClassifier(road_data_dir=road_data_dir, feature_mode=mode)
            elif mode == 'intersection_only':
                classifier = EnhancedWorkStopClassifier(intersection_data_dir=intersection_data_dir, feature_mode=mode, intersection_feature_set=intersection_feature_set_mode)
            elif mode == 'time_only':
                classifier = EnhancedWorkStopClassifier(time_data_dir=time_data_dir, feature_mode=mode)
            # elif mode == 'both':  # both
            #     classifier = EnhancedWorkStopClassifier(poi_data_dir=poi_data_dir, road_data_dir=road_data_dir, feature_mode=mode)
            elif mode == 'all':
                classifier = EnhancedWorkStopClassifier(poi_data_dir=poi_data_dir, road_data_dir=road_data_dir, intersection_data_dir=intersection_data_dir,
                                                        intersection_feature_set=intersection_feature_set_mode, time_data_dir=time_data_dir,
                                                        feature_mode=mode)
            elif mode == 'poi_time':
                classifier = EnhancedWorkStopClassifier(poi_data_dir=poi_data_dir, time_data_dir=time_data_dir,
                                                        feature_mode=mode)
            elif mode == 'road_time':
                classifier = EnhancedWorkStopClassifier(road_data_dir=road_data_dir, time_data_dir=time_data_dir,
                                                        feature_mode=mode)
            elif mode == 'intersection_time':
                classifier = EnhancedWorkStopClassifier(intersection_data_dir=intersection_data_dir,
                                                        intersection_feature_set=intersection_feature_set_mode, time_data_dir=time_data_dir,
                                                        feature_mode=mode)

            # Run pipeline
            results = classifier.run_enhanced_pipeline(min_recalls=[0.85], models=["lgbm"])
            # all_results[mode] = results

            print(f"\n🎉 {mode.upper()} classification completed successfully!")

            # comment out the run_enhanced_pipeline, 
            # and uncomment the following block to prepare_splits_only, load_trained_model, load_decision_threshold_from_metrics and run_fp_diagnosis
            # Note only one model should be load each time
            #classifier.prepare_splits_only(seed=RANDOM_STATE)
            #classifier.load_trained_model(r"C:\Users\masters_research\driver_stop_purpose_classification\output\ml_results\saved_ml_models\LGBM_ALL_all_20260111_175258.pkl")
            #classifier.load_decision_threshold_from_metrics(r"C:\Users\masters_research\driver_stop_purpose_classification\output\ml_results\saved_ml_performance_metrics\phase2_threshold_tunning_diff_seed\LGBM_ALL_all_20260111_175258_metrics_seed42_minrecall0.85.json")
            #classifier.load_trained_model(r"C:\Users\masters_research\driver_stop_purpose_classification\output\ml_results\saved_ml_models\LGBM_ALL_all_20260116_213149.pkl")
            #classifier.load_decision_threshold_from_metrics(r"C:\Users\masters_research\driver_stop_purpose_classification\output\ml_results\saved_ml_performance_metrics\phase2_threshold_tunning_diff_seed\LGBM_ALL_all_20260116_213149_metrics_seed10_minrecall0.85.json")
            #classifier.load_trained_model(r"C:\Users\masters_research\driver_stop_purpose_classification\output\ml_results\saved_ml_models\LGBM_ALL_all_20260116_222416.pkl")
            #classifier.load_decision_threshold_from_metrics(r"C:\Users\masters_research\driver_stop_purpose_classification\output\ml_results\saved_ml_performance_metrics\phase2_threshold_tunning_diff_seed\LGBM_ALL_all_20260116_222416_metrics_seed25_minrecall0.85.json")

            #classifier.run_fp_diagnosis(seed=RANDOM_STATE, out_dir=r".\fp_diagnostics_naics_nn", split="test")

        except Exception as e:
            print(f"❌ Error in {mode} mode: {e}")
            import traceback
            traceback.print_exc()

    # Compare all modes
    if all_results:
        print(f"\n{'='*80}")
        print("🏆 FINAL COMPARISON ACROSS ALL FEATURE MODES")
        print(f"{'='*80}")

        print(f"{'Feature Mode':<15} {'Best Model':<15} {'Accuracy':<12} {'ROC AUC Score':<12} {'PR AUC Score':<12}")
        print("-" * 60)

        for mode, results in all_results.items():
            if results['best_model']:
                best_metrics = results['results'][results['best_model']]
                print("best_metrics", best_metrics)
                print(f"{mode.upper():<15} {results['best_model']:<15} {best_metrics['accuracy']:<12.4f} {best_metrics['roc_auc']:<12.4f} {best_metrics['pr_auc']:<12.4f}")

        print(f"\n✅ All feature mode comparisons completed!")

    return all_results


if __name__ == "__main__":
    main()
