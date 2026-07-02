#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained XGBoost model and predict targets for a feature TSV."
    )

    parser.add_argument("--model-file", required=True)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--output-file", required=True)

    parser.add_argument("--feature-id-col", default="id")
    parser.add_argument("--target-col", required=True)

    parser.add_argument(
        "--target-sep",
        default=None,
        help="Separator for target file. If omitted, inferred from extension."
    )

    return parser.parse_args()


def infer_sep(path):
    if path.endswith(".csv"):
        return ","
    return "\t"


def load_prediction_data(args):
    features = pd.read_csv(args.feature_file, sep="\t")

    target_sep = args.target_sep
    if target_sep is None:
        target_sep = infer_sep(args.target_file)

    targets = pd.read_csv(args.target_file, sep=target_sep)

    if args.feature_id_col not in features.columns:
        raise ValueError(f"Feature ID column not found: {args.feature_id_col}")

    if args.target_col not in targets.columns:
        raise ValueError(f"Target column not found: {args.target_col}")

    if len(features) != len(targets):
        raise ValueError(
            f"Feature file and target file have different numbers of rows: "
            f"{len(features)} vs {len(targets)}"
        )

    ids = features[args.feature_id_col].values
    y_true = targets[args.target_col].values

    # Feature columns are all columns except ID.
    feature_cols = [
        col for col in features.columns
        if col != args.feature_id_col
    ]

    X = features[feature_cols]

    # XGBoost needs numeric input.
    non_numeric = X.columns[~X.dtypes.apply(lambda x: np.issubdtype(x, np.number))]
    if len(non_numeric) > 0:
        raise ValueError(
            "Non-numeric feature columns found: "
            + ", ".join(non_numeric)
        )

    return X, y_true, ids


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)

    return {
        "rmse": rmse,
        "pearson": pearson_corr,
        "pearson_p": pearson_p,
        "spearman": spearman_corr,
        "spearman_p": spearman_p,
    }


def main():
    args = parse_args()

    X, y_true, ids = load_prediction_data(args)

    model = xgb.XGBRegressor()
    model.load_model(args.model_file)

    y_pred = model.predict(X)

    metrics = compute_metrics(y_true, y_pred)

    print(f"Number of samples: {len(y_true)}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Pearson: {metrics['pearson']:.6f}")
    print(f"Pearson p-value: {metrics['pearson_p']:.6e}")
    print(f"Spearman: {metrics['spearman']:.6f}")
    print(f"Spearman p-value: {metrics['spearman_p']:.6e}")

    out = pd.DataFrame(
        {
            "id": ids,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    out.to_csv(args.output_file, sep="\t", index=False)


if __name__ == "__main__":
    main()