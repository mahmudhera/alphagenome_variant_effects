#!/usr/bin/env python3

import argparse
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost regression model from feature and target tables."
    )

    # Input/output files
    parser.add_argument("--feature-file", required=True, help="TSV file containing id and feature columns")
    parser.add_argument("--target-file", required=True, help="TSV/CSV file containing ids and target column")
    parser.add_argument("--output-dir", required=True, help="Directory to save model and predictions")

    # Column names
    parser.add_argument("--feature-id-col", default="id", help="ID column in feature file")
    parser.add_argument("--target-id-col", required=True, help="ID column in target file")
    parser.add_argument("--target-col", required=True, help="Target column in target file")

    # File format
    parser.add_argument(
        "--target-sep",
        default=None,
        help="Separator for target file. If not given, inferred from extension."
    )

    # Data splitting
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # XGBoost training parameters
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1)

    return parser.parse_args()


def infer_sep(path):
    """Infer whether the file is CSV or TSV from its extension."""
    if path.endswith(".csv"):
        return ","
    return "\t"


def load_data(args):
    """Load feature and target files, merge by id, and return X, y, ids."""

    features = pd.read_csv(args.feature_file, sep="\t")

    target_sep = args.target_sep
    if target_sep is None:
        target_sep = infer_sep(args.target_file)

    targets = pd.read_csv(args.target_file, sep=target_sep)

    if args.feature_id_col not in features.columns:
        raise ValueError(f"Feature ID column not found: {args.feature_id_col}")

    if args.target_id_col not in targets.columns:
        raise ValueError(f"Target ID column not found: {args.target_id_col}")

    if args.target_col not in targets.columns:
        raise ValueError(f"Target column not found: {args.target_col}")

    # Keep only ID and target from the target file.
    targets = targets[[args.target_id_col, args.target_col]].copy()

    # Rename target ID column so both files use the same ID name during merge.
    targets = targets.rename(columns={args.target_id_col: args.feature_id_col})

    # Merge keeps only IDs that appear in both files.
    data = features.merge(targets, on=args.feature_id_col, how="inner")

    if data.empty:
        raise ValueError("No matching IDs found between feature file and target file.")

    ids = data[args.feature_id_col].values
    y = data[args.target_col].values

    # Feature columns are all columns except ID and target.
    feature_cols = [
        col for col in data.columns
        if col not in {args.feature_id_col, args.target_col}
    ]

    X = data[feature_cols]

    # XGBoost needs numeric features.
    non_numeric = X.columns[~X.dtypes.apply(lambda x: np.issubdtype(x, np.number))]
    if len(non_numeric) > 0:
        raise ValueError(
            "Non-numeric feature columns found: "
            + ", ".join(non_numeric)
            + "\nPlease encode or remove these columns before training."
        )

    return X, y, ids, feature_cols


def split_data(X, y, ids, args):
    """Create train, validation, and test splits."""

    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
        X,
        y,
        ids,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # val_size is interpreted as a fraction of the full dataset.
    val_fraction_of_train_val = args.val_size / (1.0 - args.test_size)

    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_train_val,
        y_train_val,
        ids_train_val,
        test_size=val_fraction_of_train_val,
        random_state=args.seed,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, ids_train, ids_val, ids_test


def train_model(X_train, y_train, X_val, y_val, args):
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=args.seed,
        n_jobs=args.n_jobs,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    return model


def evaluate_model(model, X, y, split_name):
    """Evaluate RMSE on a given split."""

    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)

    print(f"{split_name} RMSE: {rmse:.6f}")

    # also compute correlation pearson and spearman
    pearson_corr = np.corrcoef(y, pred)[0, 1]
    spearman_corr = pd.Series(y).corr(pd.Series(pred), method="spearman")
    print(f"{split_name} Pearson correlation: {pearson_corr:.6f}")
    print(f"{split_name} Spearman correlation: {spearman_corr:.6f}")

    return pred, rmse, pearson_corr, spearman_corr


def save_predictions(path, ids, y_true, y_pred):
    """Save predictions with IDs."""

    out = pd.DataFrame(
        {
            "id": ids,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    out.to_csv(path, sep="\t", index=False)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    X, y, ids, feature_cols = load_data(args)

    print(f"Loaded {X.shape[0]} samples")
    print(f"Loaded {X.shape[1]} features")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        ids_train,
        ids_val,
        ids_test,
    ) = split_data(X, y, ids, args)

    print(f"Train samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    model = train_model(X_train, y_train, X_val, y_val, args)

    train_pred, train_rmse, train_pearson, train_spearman = evaluate_model(model, X_train, y_train, "Train")
    val_pred, val_rmse, val_pearson, val_spearman = evaluate_model(model, X_val, y_val, "Validation")
    test_pred, test_rmse, test_pearson, test_spearman = evaluate_model(model, X_test, y_test, "Test")

    model_path = os.path.join(args.output_dir, "xgboost_model.json")
    model.save_model(model_path)

    save_predictions(
        os.path.join(args.output_dir, "train_predictions.tsv"),
        ids_train,
        y_train,
        train_pred,
    )

    save_predictions(
        os.path.join(args.output_dir, "validation_predictions.tsv"),
        ids_val,
        y_val,
        val_pred,
    )

    save_predictions(
        os.path.join(args.output_dir, "test_predictions.tsv"),
        ids_test,
        y_test,
        test_pred,
    )

    metrics = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "rmse": [train_rmse, val_rmse, test_rmse],
            "pearson": [train_pearson, val_pearson, test_pearson],
            "spearman": [train_spearman, val_spearman, test_spearman],
        }
    )

    metrics.to_csv(
        os.path.join(args.output_dir, "metrics.tsv"),
        sep="\t",
        index=False,
    )

    feature_info = pd.DataFrame({"feature": feature_cols})
    feature_info.to_csv(
        os.path.join(args.output_dir, "features_used.tsv"),
        sep="\t",
        index=False,
    )

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()