from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .features import add_color_indices, get_feature_columns
from .quantile_binner import QuantileBinner
from .weighted_hist_nb import WeightedHistogramNB


def run_experiment(
    df: pd.DataFrame,
    target: str,
    use_color_features: bool,
    redshift_weight: float,
    uniform_prior: bool,
    n_bins: int,
    test_size: float,
    random_state: int,
):
    work = add_color_indices(df) if use_color_features else df.copy()
    features = get_feature_columns(use_color_features=use_color_features)

    missing = [col for col in features + [target] if col not in work.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = work[features].copy()
    y = work[target].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    binner = QuantileBinner(n_bins=n_bins)
    X_train_bin = binner.fit_transform(X_train)
    X_test_bin = binner.transform(X_test)

    weights = {col: 1.0 for col in features}
    if "redshift" in weights:
        weights["redshift"] = redshift_weight

    model = WeightedHistogramNB(
        alpha=1.0,
        uniform_prior=uniform_prior,
        feature_weights=weights,
    )
    model.fit(X_train_bin, y_train)
    pred = model.predict(X_test_bin)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "report": classification_report(y_test, pred, digits=4),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/star_classification.csv")
    parser.add_argument("--target", type=str, default="class")
    parser.add_argument("--n-bins", type=int, default=192)
    parser.add_argument("--redshift-weight", type=float, default=2.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Please put the CSV file under data/ first."
        )

    df = pd.read_csv(data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "model": "weighted_hist_nb",
            "use_color_features": True,
            "redshift_weight": args.redshift_weight,
            "uniform_prior": True,
        },
        {
            "model": "no_color_features",
            "use_color_features": False,
            "redshift_weight": args.redshift_weight,
            "uniform_prior": True,
        },
        {
            "model": "no_redshift_weight",
            "use_color_features": True,
            "redshift_weight": 1.0,
            "uniform_prior": True,
        },
        {
            "model": "empirical_prior",
            "use_color_features": True,
            "redshift_weight": args.redshift_weight,
            "uniform_prior": False,
        },
    ]

    rows = []
    main_report = ""
    for exp in experiments:
        metrics = run_experiment(
            df=df,
            target=args.target,
            use_color_features=exp["use_color_features"],
            redshift_weight=exp["redshift_weight"],
            uniform_prior=exp["uniform_prior"],
            n_bins=args.n_bins,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        rows.append(
            {
                "model": exp["model"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "redshift_weight": exp["redshift_weight"],
                "use_color_features": exp["use_color_features"],
                "uniform_prior": exp["uniform_prior"],
            }
        )
        if exp["model"] == "weighted_hist_nb":
            main_report = metrics["report"]

    result_df = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    result_df.to_csv(out_dir / "ablation.csv", index=False)
    result_df.head(1).to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "classification_report.txt").write_text(main_report, encoding="utf-8")

    print("Saved results to", out_dir)
    print(result_df)


if __name__ == "__main__":
    main()
