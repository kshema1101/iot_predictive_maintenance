"""
ML Training Pipeline for Predictive Maintenance

Trains three models from the generated dataset:
  1. Random Forest Classifier  → failure mode classification (with hyperparameter tuning)
  2. XGBoost Regressor         → Remaining Useful Life (RUL) prediction
  3. LSTM Neural Network        → time-series forecasting (next N cycles)

All models are saved to ml/models/ as .joblib or .keras files.

Usage:
  python ml/train_models.py                           # train all models
  python ml/train_models.py --data ml/data/training_data.csv
  python ml/train_models.py --skip-lstm               # skip LSTM (faster)
  python ml/train_models.py --no-tune                 # skip hyperparameter tuning
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"

FEATURE_COLS = [
    "motor_current", "transition_time", "vibration_peak",
    "supply_voltage", "motor_temperature",
    "current_x_time", "vibration_x_current", "power_draw",
]

SEQUENCE_FEATURES = [
    "motor_current", "transition_time", "vibration_peak",
    "supply_voltage", "motor_temperature",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples from {path}")
    print(f"Failure modes: {df['failure_mode'].value_counts().to_dict()}")
    return df


def _filter_degraded_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only samples where degradation is actually observable.
    For failing switches, drop early cycles where sensor readings are
    indistinguishable from healthy — these mislabeled samples are the
    main cause of low accuracy.
    """
    healthy = df[df["failure_mode"] == "healthy"]
    failing = df[df["failure_mode"] != "healthy"]
    failing_visible = failing[failing["degradation_progress"] >= 0.05]

    combined = pd.concat([healthy, failing_visible], ignore_index=True)
    dropped = len(df) - len(combined)
    print(f"  Filtered out {dropped} early-cycle samples (indistinguishable from healthy)")
    print(f"  Training on {len(combined)} samples with observable degradation signatures")
    return combined


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Model 1: Failure Mode Classification (RF + XGBoost, pick best)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_failure_classifier(df: pd.DataFrame, tune: bool = True):
    """
    Trains both Random Forest and XGBoost classifiers, tunes hyperparameters,
    and saves the best one.
    Input:  5 sensor features + 3 engineered features
    Output: failure_mode (healthy, mechanical_friction, blockage, electrical, bearing_wear)
    """
    print("\n" + "=" * 60)
    print("  MODEL 1: Failure Classification (RF vs XGBoost)")
    print("=" * 60)

    df_filtered = _filter_degraded_samples(df)

    X = df_filtered[FEATURE_COLS].values
    y = df_filtered["failure_mode"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_name = ""
    best_score = 0

    # ── Candidate 1: Random Forest with optional tuning ──────────
    print("\n  Training Random Forest...")
    if tune:
        rf_param_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            rf_param_grid,
            n_iter=30,
            cv=cv,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        rf_search.fit(X_train, y_train)
        rf_model = rf_search.best_estimator_
        print(f"  Best RF params: {rf_search.best_params_}")
        print(f"  Best RF CV F1 (macro): {rf_search.best_score_:.4f}")
    else:
        rf_model = RandomForestClassifier(
            n_estimators=400, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)

    rf_cv_scores = cross_val_score(rf_model, X_scaled, y_encoded, cv=cv, scoring="f1_macro")
    rf_score = rf_cv_scores.mean()
    print(f"  RF 5-fold CV F1: {rf_score:.4f} (±{rf_cv_scores.std():.4f})")

    if rf_score > best_score:
        best_score = rf_score
        best_model = rf_model
        best_name = "Random Forest"

    # ── Candidate 2: XGBoost Classifier with optional tuning ─────
    print("\n  Training XGBoost Classifier...")
    if tune:
        xgb_param_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0.5, 1.0, 2.0],
        }
        xgb_search = RandomizedSearchCV(
            xgb.XGBClassifier(
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, n_jobs=-1, use_label_encoder=False,
            ),
            xgb_param_grid,
            n_iter=30,
            cv=cv,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        xgb_search.fit(X_train, y_train)
        xgb_clf = xgb_search.best_estimator_
        print(f"  Best XGB params: {xgb_search.best_params_}")
        print(f"  Best XGB CV F1 (macro): {xgb_search.best_score_:.4f}")
    else:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, n_jobs=-1, use_label_encoder=False,
        )
        xgb_clf.fit(X_train, y_train)

    xgb_cv_scores = cross_val_score(xgb_clf, X_scaled, y_encoded, cv=cv, scoring="f1_macro")
    xgb_score = xgb_cv_scores.mean()
    print(f"  XGB 5-fold CV F1: {xgb_score:.4f} (±{xgb_cv_scores.std():.4f})")

    if xgb_score > best_score:
        best_score = xgb_score
        best_model = xgb_clf
        best_name = "XGBoost"

    # ── Evaluate the winner on held-out test set ─────────────────
    print(f"\n  Winner: {best_name} (CV F1 = {best_score:.4f})")
    print("-" * 60)

    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    labels = le.classes_
    header = "              " + "  ".join(f"{l[:8]:>8s}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:8d}" for v in row)
        print(f"  {labels[i]:12s}  {row_str}")

    if hasattr(best_model, "feature_importances_"):
        importances = sorted(
            zip(FEATURE_COLS, best_model.feature_importances_), key=lambda x: -x[1]
        )
        print("\nFeature Importance:")
        for feat, imp in importances:
            bar = "█" * int(imp * 50)
            print(f"  {feat:25s} {imp:.3f} {bar}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "failure_classifier_rf.joblib")
    joblib.dump(le, MODEL_DIR / "failure_label_encoder.joblib")
    joblib.dump(scaler, MODEL_DIR / "failure_scaler.joblib")
    print(f"\nSaved → {MODEL_DIR}/failure_classifier_rf.joblib ({best_name})")

    return best_model, le, scaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Model 2: XGBoost — Remaining Useful Life (RUL) Regression
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_rul_predictor(df: pd.DataFrame):
    """
    XGBoost Regressor
    Input:  sensor features + engineered features + degradation_progress
    Output: remaining_useful_life (number of cycles until failure)
    """
    print("\n" + "=" * 60)
    print("  MODEL 2: XGBoost — Remaining Useful Life (RUL)")
    print("=" * 60)

    df_failing = df[df["failure_mode"] != "healthy"].copy()
    print(f"Training on {len(df_failing)} samples (failing switches only)")

    rul_features = FEATURE_COLS + ["degradation_progress"]
    X = df_failing[rul_features].values
    y = df_failing["remaining_useful_life"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nRegression Metrics:")
    print(f"  MAE  (Mean Absolute Error):  {mae:.2f} cycles")
    print(f"  RMSE (Root Mean Squared):    {rmse:.2f} cycles")
    print(f"  R²   (Explained Variance):   {r2:.4f}")

    importances = sorted(
        zip(rul_features, model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nFeature Importance:")
    for feat, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"  {feat:25s} {imp:.3f} {bar}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "rul_predictor_xgb.joblib")
    joblib.dump(scaler, MODEL_DIR / "rul_scaler.joblib")
    print(f"\nSaved → {MODEL_DIR}/rul_predictor_xgb.joblib")

    return model, scaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Model 3: LSTM — Time-Series Forecasting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_sequences(df: pd.DataFrame, window: int = 10, horizon: int = 5):
    """
    Build sliding window sequences grouped by switch_id.
    X shape: (samples, window, features)
    y shape: (samples, horizon, features)
    """
    X_all, y_all = [], []

    for switch_id, group in df.groupby("switch_id"):
        group = group.sort_values("cycle")
        values = group[SEQUENCE_FEATURES].values

        for i in range(len(values) - window - horizon + 1):
            X_all.append(values[i : i + window])
            y_all.append(values[i + window : i + window + horizon])

    return np.array(X_all), np.array(y_all)


def train_lstm_forecaster(df: pd.DataFrame, window: int = 10, horizon: int = 5):
    """
    LSTM Network
    Input:  sequence of last `window` telemetry readings (10 × 5 features)
    Output: predicted next `horizon` readings (5 × 5 features)
    """
    print("\n" + "=" * 60)
    print("  MODEL 3: LSTM — Time-Series Forecasting")
    print("=" * 60)

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("TensorFlow not installed. Skipping LSTM training.")
        print("Install with: pip install tensorflow")
        return None, None

    X, y = build_sequences(df, window, horizon)
    print(f"Sequences: X={X.shape}, y={y.shape}")

    n_features = X.shape[2]

    scaler = StandardScaler()
    X_flat = X.reshape(-1, n_features)
    scaler.fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(X.shape)
    y_scaled = scaler.transform(y.reshape(-1, n_features)).reshape(y.shape)

    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    model = keras.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.RepeatVector(horizon),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(0.2),
        layers.TimeDistributed(layers.Dense(n_features)),
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
        ],
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, n_features)).reshape(y_pred_scaled.shape)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, n_features)).reshape(y_test.shape)

    print(f"\nForecasting Metrics (per feature, averaged over {horizon}-step horizon):")
    for i, feat in enumerate(SEQUENCE_FEATURES):
        mae = mean_absolute_error(y_actual[:, :, i].flatten(), y_pred[:, :, i].flatten())
        print(f"  {feat:25s} MAE = {mae:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "lstm_forecaster.keras")
    joblib.dump(scaler, MODEL_DIR / "lstm_scaler.joblib")
    print(f"\nSaved → {MODEL_DIR}/lstm_forecaster.keras")

    return model, scaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="Train Predictive Maintenance ML Models")
    parser.add_argument("--data", default=str(DATA_DIR / "training_data.csv"),
                        help="Path to training CSV")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM training (requires TensorFlow)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning (faster, uses sensible defaults)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print("Run 'python ml/generate_dataset.py' first to create it.")
        return

    df = load_data(str(data_path))

    train_failure_classifier(df, tune=not args.no_tune)
    train_rul_predictor(df)

    if not args.skip_lstm:
        train_lstm_forecaster(df)
    else:
        print("\nSkipped LSTM training (--skip-lstm)")

    print("\n" + "=" * 60)
    print("  ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nModel files saved to: {MODEL_DIR}/")
    for f in sorted(MODEL_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
