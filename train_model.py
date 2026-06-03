from pathlib import Path

import argparse
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DEFAULT_INPUT_CSV = Path("synthetic_bengaluru_traffic.csv")
DEFAULT_MODEL_PATH = Path("xgboost_traffic_model.pkl")
DEFAULT_METRICS_PATH = Path("xgboost_traffic_model_metrics.json")
TARGET_COLUMN = "target_speed_kmh"
DROP_COLUMNS = ["edge_id"]
CATEGORICAL_FEATURES = ["highway_type", "weather"]
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost traffic-speed model from synthetic Bengaluru traffic data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Training CSV path. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Saved model path. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Saved metrics JSON path. Default: {DEFAULT_METRICS_PATH}",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing. Default: 0.2",
    )
    return parser.parse_args()


def load_training_data(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Generate synthetic_bengaluru_traffic.csv first."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")

    required_columns = set(CATEGORICAL_FEATURES + [TARGET_COLUMN])
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Training CSV is missing required column(s): {missing}")

    return df


def build_features_and_target(df):
    columns_to_drop = [TARGET_COLUMN] + [col for col in DROP_COLUMNS if col in df.columns]
    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN]

    missing_categorical = set(CATEGORICAL_FEATURES) - set(X.columns)
    if missing_categorical:
        missing = ", ".join(sorted(missing_categorical))
        raise ValueError(f"Feature data is missing categorical column(s): {missing}")

    return X, y


def build_model_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
    )

    regressor = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def evaluate_model(model_pipeline, X_test, y_test):
    predictions = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)

    return {
        "mae_kmh": round(float(mae), 4),
        "rmse_kmh": round(float(rmse), 4),
        "r2_score": round(float(r2), 4),
    }


def save_outputs(model_pipeline, metrics, model_path, metrics_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pipeline, model_path, compress=3)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    args = parse_args()

    print("1. Loading synthetic traffic data...")
    df = load_training_data(args.input)
    print(f"   Loaded {len(df):,} rows from {args.input}")

    print("2. Preparing features and target...")
    X, y = build_features_and_target(df)
    print(f"   Training features: {', '.join(X.columns)}")

    print("3. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )
    print(f"   Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")

    print("4. Building the machine learning pipeline...")
    model_pipeline = build_model_pipeline()

    print("5. Training the XGBoost model...")
    model_pipeline.fit(X_train, y_train)

    print("6. Evaluating model accuracy...")
    metrics = evaluate_model(model_pipeline, X_test, y_test)
    print(
        "   Model Performance: "
        f"MAE {metrics['mae_kmh']:.2f} km/h | "
        f"RMSE {metrics['rmse_kmh']:.2f} km/h | "
        f"R2 {metrics['r2_score']:.3f}"
    )

    print("7. Saving trained model and metrics...")
    save_outputs(model_pipeline, metrics, args.model_output, args.metrics_output)
    print(f"   Model saved as '{args.model_output}'")
    print(f"   Metrics saved as '{args.metrics_output}'")


if __name__ == "__main__":
    main()