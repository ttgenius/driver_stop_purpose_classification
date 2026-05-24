#!/usr/bin/env python3
"""
Baseline Work Stop Classification for Chapter 4 Primary Evaluation

This script is for running the Chapter 4 primary evaluation in Yuezhang Zhu's Master's thesis:
1. Evaluates multiple machine learning models on the baseline dataset
for the task of classifying driver stops as work stops or non-work stops.
Models: 
Logistic Regression
Random Forest
LightGBM, 
multi-layer perceptron
convolutional neural network
Transformer-inspired neural network

2. Compare model perfromanc across different feature combinations to examine 
the contribution of contextual information to stop purpose classification. 
Feature combindations exmained: 
time
time + poi
time + road
time + intersection
all = time + poi + road + intersection

The results establish baseline model performance and identify key modelling challenges 
that motivate the subsequent experiments in this thesis.
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
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc
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

REPO_ROOT_DIR = Path(__file__).resolve().parents[2]
BASE_INPUT_DIR = REPO_ROOT_DIR / "input"
BASE_OUTPUT_DIR = REPO_ROOT_DIR / "output"
RANDOM_STATE = 10

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



class BaselineWorkStopClassifier:
    """Baseline classifier supporting the Chapter 4 feature combinations."""
    
    def __init__(self, poi_data_dir=None, road_data_dir=None, intersection_data_dir=None, intersection_feature_set='core',
                 time_data_dir=None, feature_mode='poi_time'):
        """
        Initialize the baseline classifier.
        
        Args:
            poi_data_dir (str, optional): Path to POI feature data directory
            road_data_dir (str, optional): Path to road feature data directory
            intersection_data_dir (str, optional): Path to intersection feature data directory
            intersection_feature_set (str): Intersection feature subset to use
            time_data_dir (str, optional): Path to time feature data directory
            feature_mode (str): Feature combination to train on.
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
        # elif self.feature_mode == 'both' and (not self.poi_data_dir or not self.road_data_dir):
        #     raise ValueError("Both poi_data_dir and road_data_dir are required when feature_mode is 'both'")
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
        
        print(f"🎯 Initialized Baseline Classifier with feature mode: '{self.feature_mode.upper()}'")
        # if self.feature_mode == 'poi_only':
        #     print(f"   📍 POI data: {self.poi_data_dir}")
        # elif self.feature_mode == 'road_only':
        #     print(f"   🛣️  Road data: {self.road_data_dir}")
        # elif self.feature_mode == 'time_only':
        #     print(f"   🛣️  time data: {self.time_data_dir}")
        # elif self.feature_mode =='poi_time':
        #     print(f"   📍 POI data: {self.poi_data_dir}")
        #     print(f"   🛣️  time data: {self.time_data_dir}")
        # else:  # both
        # else:  # both
        #     print(f"   📍 POI data: {self.poi_data_dir}")
        #     print(f"   🛣️  Road data: {self.road_data_dir}")

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
                               test_loss=None, test_accuracy=None):
        """Save trained model and performance metrics to files"""

        # Create timestamp
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

            # Save metrics
            metrics_path = metrics_dir / f"{model_filename}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            print(f"   📊 Metrics saved: {metrics_path}")

            return model_path, metrics_path

        except Exception as e:
            print(f"   ❌ Error saving model/metrics: {e}")
            return None, None
    
    def load_and_merge_data(self, num_files=5):
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
            return self._load_all_data(num_files)
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

    def _load_all_data(self, num_files):
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
                matched_files.append((poi_file, road_file, intersection_file, time_file))

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
        for i, (poi_file, road_file, intersection_file, time_file) in enumerate(files_to_load, 1):
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

            # if i==1:
            #     merged_df.head(50).to_csv('merged_data_50.csv')
            #     print("output to merged_data_50.csv")

            dataframes.append(merged_df)


        # Combine all dataframes
        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(self.data)} total records with {self.data.shape[1]} columns (POI + road + intersection + time)")

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
        # elif self.feature_mode == 'both':  # both
        #     selected_features = all_feature_columns
        #     print(f"Selected {len(selected_features)} total features:")
        #     print(f"   POI features: {len(poi_features)}")
        #     print(f"   Road features: {len(road_features)}")
        #     print(f"   Other features: {len(selected_features) - len(poi_features) - len(road_features)}")
        elif self.feature_mode == 'all':
            selected_features = list(set(poi_features + road_features + intersection_features + time_features))
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
        
        # Prepare data
        self.features = self.data[selected_features].fillna(-1)
        self.target = self.data['is_work_stop']
        self.groups = self.data['unit_id']
        #self.groups = self.data['naics1_long']
        self.feature_names = selected_features
        
        print(f"\nDataset info:")
        print(f"   Samples: {len(self.features)}")
        print(f"   Features: {self.features.shape[1]}")
        print(f"   Work stops: {(self.target == 1).sum()}")
        print(f"   Non-work stops: {(self.target == 0).sum()}")
        print(f"   Unique vehicles: {self.groups.nunique()}")
        
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

        return X_train, X_val, X_test, y_train, y_val, y_test


    def train_random_forest(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train Random Forest with validation-based early stopping"""

        print(f"\n🌳 Training Random Forest ({self.feature_mode.upper()} features)...")

        model_name = f'RF_{self.feature_mode.upper()}'

        # -------------------------------
        # Early stopping configuration
        # -------------------------------
        max_estimators = 600
        step = 50
        patience = 2
        best_val_auc = -np.inf
        patience_counter = 0

        # -------------------------------
        # Initialize RF with warm_start
        # -------------------------------
        rf = RandomForestClassifier(
            n_estimators=0,
            max_depth=12,
            min_samples_split=100,
            min_samples_leaf=50,
            max_features='sqrt',
            warm_start=True,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        start_time = time.time()
        best_rf = None
        best_n_estimators = 0

        # -------------------------------
        # Incremental training loop
        # -------------------------------
        for n_estimators in range(step, max_estimators + 1, step):
            rf.set_params(n_estimators=n_estimators)
            rf.fit(X_train, y_train)

            val_pred_proba = rf.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred_proba)

            print(f"   🌱 Trees: {n_estimators:3d} | Val ROC-AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_rf = deepcopy(rf)  # 🔑 CRITICAL FIX
                best_n_estimators = n_estimators
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"   🛑 Early stopping triggered at {n_estimators} trees")
                break

        training_time = time.time() - start_time

        # -------------------------------
        # Use BEST model
        # -------------------------------
        rf = best_rf

        # -------------------------------
        # Test evaluation
        # -------------------------------
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]

        metrics = self.calculate_comprehensive_metrics(
            y_test, y_pred, y_pred_proba, model_name
        )

        self.print_comprehensive_metrics(
            metrics, f"Random Forest ({self.feature_mode.upper()})"
        )

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
            "best_validation_roc_auc": best_val_auc
        }
        # -------------------------------
        # Save model & metrics
        # -------------------------------
        self.save_model_and_metrics(
            rf,
            model_name,
            metrics,
            training_time,
            BASE_OUTPUT_DIR,
            feature_names=self.feature_names,
            hyperparameters=rf_hyperparameters,
            feature_importance=rf.feature_importances_
        )

        # -------------------------------
        # Feature importance
        # -------------------------------
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
    
    def train_neural_network(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train Neural Network with selected features"""
        
        print(f"\n🧠 Training Neural Network ({self.feature_mode.upper()} features)...")
        
        # Check GPU availability for NN training
        if gpu_available:
            print("   🎮 Attempting GPU acceleration for Neural Network training")
        else:
            print("   💻 Using optimized CPU for Neural Network training")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Neural network architecture
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
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        epoches = 100
        batch_size = 2048

        start_time = time.time()
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=epoches,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val),
            verbose=0,
            callbacks=[early_stopping, reduce_lr, EpochLogger(log_every=5)]
        )

        # Calculate training time
        training_time = time.time() - start_time

        # Predictions
        y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate comprehensive metrics
        model_name = f'NN_{self.feature_mode.upper()}'
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba, model_name)

        # Print comprehensive metrics
        self.print_comprehensive_metrics(metrics, f"Neural Network ({self.feature_mode.upper()})")
        print(f"   ⏱️  Training time: {training_time:.2f} seconds")

        nn_hyperparameters = {
            "model_type": "NN",
            "input_dim": X_train.shape[1],
            "layers":[['Dense 128 relu', 'BatchNormalization', 'Dropout 0.4'], ['Dense 64 relu', 'BatchNormalization', 'Dropout 0.3'], ['Dense 1 sigmoid']],
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "batch_size": batch_size,
            "epochs": epoches,
            "loss":"binary_crossentropy",
            "early_stopping": 10,
            "reduced_lr": 0.2
        }

        test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

        # Save model and metrics
        self.save_model_and_metrics(model, model_name, metrics, training_time, hyperparameters=nn_hyperparameters,
                                    training_history=history, test_loss=test_loss, test_accuracy=test_accuracy)
        
        # Store model and results
        self.models[model_name] = model
        self.results[model_name] = metrics
        
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

    def train_lightgbm(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train LightGBM with validation + early stopping"""

        print(f"\n💡 Training LightGBM ({self.feature_mode.upper()} features)...")
        model_name = f"LGBM_{self.feature_mode.upper()}"

        # Sanitize feature names
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        X_train.columns = X_train.columns.str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)
        X_val.columns = X_train.columns
        X_test.columns = X_train.columns

        lgbm = lgb.LGBMClassifier(
            n_estimators=1000,  # upper bound
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=50,
            colsample_bytree=0.8,
            subsample=0.8,
            early_stopping_rounds=50,
            objective='binary',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )

        start_time = time.time()

        lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
        )

        training_time = time.time() - start_time

        # ✅ Use best iteration automatically
        y_pred = lgbm.predict(X_test)
        y_pred_proba = lgbm.predict_proba(X_test)[:, 1]

        metrics = self.calculate_comprehensive_metrics(
            y_test, y_pred, y_pred_proba, model_name
        )

        self.print_comprehensive_metrics(metrics, f"LightGBM ({self.feature_mode.upper()})")
        print(f"   🌳 Best iteration: {lgbm.best_iteration_}")
        print(f"   ⏱️ Training time: {training_time:.2f}s")

        self.save_model_and_metrics(
            lgbm,
            model_name,
            metrics,
            training_time,
            hyperparameters={
                **lgbm.get_params(),
                "best_iteration": lgbm.best_iteration_
            },
            feature_names=self.feature_names,
            feature_importance=lgbm.feature_importances_
        )

        self.models[model_name] = lgbm
        self.results[model_name] = metrics

        return lgbm


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
    
    def run_baseline_pipeline(self, models=None):
        """Run the complete baseline classification pipeline."""
        
        # Load and merge data
        if models is None:
            models = ['rf', 'lr', 'nn', 'cnn', 'lgbm', 'tbt']

        self.load_and_merge_data(num_files=174)  # Start with 3 files for testing
        
        # Prepare features
        poi_features, road_features, intersection_features, time_features = self.prepare_features()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()

        feature_importance = None

        # Train models
        if 'rf' in models:
            _, feature_importance = self.train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
        if 'lr' in models:
            self.train_logistic_regression(X_train, X_test, y_train, y_test)
        if 'nn' in models:
            self.train_neural_network(X_train, X_val, X_test, y_train, y_val, y_test)
        if 'cnn' in models:
            self.train_cnn(X_train, X_val, X_test, y_train, y_val, y_test)
        if 'lgbm' in models:
            self.train_lightgbm(X_train, X_val, X_test, y_train, y_val, y_test)
        if 'tbt' in models:
            self.train_tabtransformer(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Compare results
        best_model = self.compare_models()
        
        print(f"\n✅ PIPELINE COMPLETED ({self.feature_mode.upper()} FEATURES)!")
        print("=" * 70)
        
        return {
            'feature_mode': self.feature_mode,
            'models': self.models,
            'results': self.results,
            'feature_importance': feature_importance,
            'best_model': best_model,
            'poi_features': poi_features,
            'road_features': road_features,
            'intersection_features': intersection_features,
            'time_features': time_features,
            'selected_features': self.feature_names
        }

def main():
    """Run the configured Chapter 4 baseline feature-mode evaluation."""
    
    print("🚀 BASELINE WORK STOP CLASSIFICATION")
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
                classifier = BaselineWorkStopClassifier(poi_data_dir=poi_data_dir, feature_mode=mode)
            elif mode == 'road_only':
                classifier = BaselineWorkStopClassifier(road_data_dir=road_data_dir, feature_mode=mode)
            elif mode == 'intersection_only':
                classifier = BaselineWorkStopClassifier(intersection_data_dir=intersection_data_dir, feature_mode=mode, intersection_feature_set=intersection_feature_set_mode)
            elif mode == 'time_only':
                classifier = BaselineWorkStopClassifier(time_data_dir=time_data_dir, feature_mode=mode)
            # elif mode == 'both':  # both
            #     classifier = BaselineWorkStopClassifier(poi_data_dir=poi_data_dir, road_data_dir=road_data_dir, feature_mode=mode)
            elif mode == 'all':
                classifier = BaselineWorkStopClassifier(poi_data_dir=poi_data_dir, road_data_dir=road_data_dir, intersection_data_dir=intersection_data_dir,
                                                        intersection_feature_set=intersection_feature_set_mode, time_data_dir=time_data_dir,
                                                        feature_mode=mode)
            elif mode == 'poi_time':
                classifier = BaselineWorkStopClassifier(poi_data_dir=poi_data_dir, time_data_dir=time_data_dir,
                                                        feature_mode=mode)
            elif mode == 'road_time':
                classifier = BaselineWorkStopClassifier(road_data_dir=road_data_dir, time_data_dir=time_data_dir,
                                                        feature_mode=mode)
            elif mode == 'intersection_time':
                classifier = BaselineWorkStopClassifier(intersection_data_dir=intersection_data_dir,
                                                        intersection_feature_set=intersection_feature_set_mode, time_data_dir=time_data_dir,
                                                        feature_mode=mode)

            # Run pipeline
            results = classifier.run_baseline_pipeline(models=['lgbm'])
            all_results[mode] = results
            
            print(f"\n🎉 {mode.upper()} classification completed successfully!")

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
