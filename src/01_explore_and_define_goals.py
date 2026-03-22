from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


GOAL_CANDIDATES: Dict[str, List[str]] = {
    "goal_schedule_setup": [
        "Scheduling.Shift.Created",
        "Scheduling.Template.ApplyModal.Applied",
    ],
    "goal_schedule_visibility": [
        "Mobile.Schedule.Loaded",
        "Shift.View.Opened",
        "ShiftDetails.View.Opened",
    ],
    "goal_time_tracking_started": [
        "PunchClock.PunchedIn",
        "PunchClock.PunchedOut",
    ],
    "goal_ops_approval": [
        "Scheduling.Shift.AssignmentChanged",
        "Scheduling.Shift.Approved",
        "Scheduling.OpenShiftRequest.Created",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/da_task.csv"),
        help="Path to trial event CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Folder for outputs",
    )
    return parser.parse_args()


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    expected = {
        "organization_id",
        "activity_name",
        "timestamp",
        "converted",
        "converted_at",
        "trial_start",
        "trial_end",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    for col in ["timestamp", "converted_at", "trial_start", "trial_end"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    df["converted"] = (
        df["converted"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    )
    if df["converted"].isna().any():
        raise ValueError("Unexpected values found in converted")
    df["converted"] = df["converted"].astype(int)

    before_dups = len(df)
    df = df.drop_duplicates().copy()
    after_dups = len(df)

    df = df.dropna(subset=["organization_id", "activity_name", "timestamp", "trial_start", "trial_end"])

    df["in_trial_window"] = (df["timestamp"] >= df["trial_start"]) & (df["timestamp"] <= df["trial_end"])
    df = df.loc[df["in_trial_window"]].copy()

    df["trial_day"] = (df["timestamp"] - df["trial_start"]).dt.days
    df["event_date"] = df["timestamp"].dt.date

    print(f"rows before dedupe: {before_dups}")
    print(f"rows after dedupe: {after_dups}")
    print(f"rows after filters: {len(df)}")
    print(f"organizations: {df['organization_id'].nunique()}")

    return df


def build_org_features(df: pd.DataFrame) -> pd.DataFrame:
    org_base = (
        df.groupby("organization_id", as_index=False)
        .agg(
            converted=("converted", "max"),
            trial_start=("trial_start", "min"),
            trial_end=("trial_end", "max"),
            converted_at=("converted_at", "max"),
            total_events=("activity_name", "size"),
            unique_activities=("activity_name", "nunique"),
            active_days=("event_date", "nunique"),
        )
    )

    event_counts = (
        df.assign(event_count=1)
        .pivot_table(
            index="organization_id",
            columns="activity_name",
            values="event_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    features = org_base.merge(event_counts, on="organization_id", how="left")

    activity_cols = [c for c in features.columns if c not in org_base.columns]
    for c in activity_cols:
        features[f"did__{c}"] = (features[c] > 0).astype(int)

    did_cols = [c for c in features.columns if c.startswith("did__")]
    features["module_breadth"] = features[did_cols].sum(axis=1)

    return features


def activity_driver_table(features: pd.DataFrame) -> pd.DataFrame:
    did_cols = [c for c in features.columns if c.startswith("did__")]
    rows = []
    for did_col in did_cols:
        activity = did_col.replace("did__", "")
        exposed = features[did_col] == 1
        not_exposed = features[did_col] == 0

        exp_conv = int(((features["converted"] == 1) & exposed).sum())
        exp_not = int(((features["converted"] == 0) & exposed).sum())
        nexp_conv = int(((features["converted"] == 1) & not_exposed).sum())
        nexp_not = int(((features["converted"] == 0) & not_exposed).sum())

        table = [[exp_conv, exp_not], [nexp_conv, nexp_not]]
        odds_ratio, p_value = fisher_exact(table)

        conv_rate_exposed = exp_conv / max(exp_conv + exp_not, 1)
        conv_rate_unexposed = nexp_conv / max(nexp_conv + nexp_not, 1)
        lift = conv_rate_exposed / max(conv_rate_unexposed, 1e-9)

        rows.append(
            {
                "activity_name": activity,
                "orgs_with_activity": int(exposed.sum()),
                "coverage_rate": exposed.mean(),
                "conv_rate_with_activity": conv_rate_exposed,
                "conv_rate_without_activity": conv_rate_unexposed,
                "lift": lift,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_adj_bh"] = multipletests(out["p_value"].values, method="fdr_bh")[1]
    out = out.sort_values(["p_value_adj_bh", "lift"], ascending=[True, False])
    return out


def model_logistic_coefficients(features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        c
        for c in features.columns
        if c not in {"organization_id", "converted", "trial_start", "trial_end", "converted_at"}
    ]

    X = features[feature_cols].copy()
    y = features["converted"].copy()

    keep = X.nunique(dropna=False) > 1
    X = X.loc[:, keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
        solver="liblinear",
    )
    clf.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"logit train auc: {train_auc:.4f}")
    print(f"logit test auc: {test_auc:.4f}")

    coef = pd.DataFrame(
        {
            "feature": X.columns,
            "coef_log_odds": clf.coef_.ravel(),
        }
    )
    coef["odds_ratio"] = np.exp(coef["coef_log_odds"])
    coef["abs_coef"] = np.abs(coef["coef_log_odds"])
    return coef.sort_values("abs_coef", ascending=False)


def model_importance(features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        c
        for c in features.columns
        if c not in {"organization_id", "converted", "trial_start", "trial_end", "converted_at"}
    ]

    X = features[feature_cols].copy()
    y = features["converted"].copy()

    # Drop flat columns before training.
    keep = X.nunique(dropna=False) > 1
    X = X.loc[:, keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=700,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"rf train auc: {train_auc:.4f}")
    print(f"rf test auc: {test_auc:.4f}")

    coef = pd.DataFrame(
        {
            "feature": X.columns,
            "feature_importance": clf.feature_importances_,
            "abs_coef": np.abs(clf.feature_importances_),
        }
    ).sort_values("abs_coef", ascending=False)

    return coef


def engagement_segmentation(features: pd.DataFrame) -> pd.DataFrame:
    seg = features[
        [
            "organization_id",
            "converted",
            "total_events",
            "active_days",
            "unique_activities",
            "module_breadth",
        ]
    ].copy()

    # Percentile bins handle repeated values cleanly.
    seg["active_days_segment"] = pd.cut(
        seg["active_days"].rank(method="average", pct=True),
        bins=[0.0, 0.33, 0.67, 1.0],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )
    seg["module_breadth_segment"] = pd.cut(
        seg["module_breadth"].rank(method="average", pct=True),
        bins=[0.0, 0.33, 0.67, 1.0],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )

    seg_out = (
        seg.groupby(["active_days_segment", "module_breadth_segment"], dropna=False, as_index=False)
        .agg(
            organizations=("organization_id", "nunique"),
            conversion_rate=("converted", "mean"),
            median_events=("total_events", "median"),
            median_active_days=("active_days", "median"),
            median_unique_activities=("unique_activities", "median"),
        )
        .sort_values(["active_days_segment", "module_breadth_segment"])
    )

    return seg_out


def bootstrap_lift_stability(
    features: pd.DataFrame,
    drivers: pd.DataFrame,
    top_k: int = 15,
    n_boot: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    top_activities = drivers.sort_values(["p_value_adj_bh", "lift"], ascending=[True, False]).head(top_k)

    rows = []
    for activity in top_activities["activity_name"]:
        did_col = f"did__{activity}"
        if did_col not in features.columns:
            continue

        lifts = []
        for _ in range(n_boot):
            sample_idx = rng.integers(0, len(features), len(features))
            sample = features.iloc[sample_idx]

            exposed = sample[did_col] == 1
            exp_total = int(exposed.sum())
            nexp_total = int((~exposed).sum())
            if exp_total == 0 or nexp_total == 0:
                continue

            exp_conv = sample.loc[exposed, "converted"].mean()
            nexp_conv = sample.loc[~exposed, "converted"].mean()
            lifts.append(exp_conv / max(nexp_conv, 1e-9))

        if not lifts:
            continue

        arr = np.array(lifts)
        rows.append(
            {
                "activity_name": activity,
                "bootstrap_n": int(arr.size),
                "lift_mean": float(arr.mean()),
                "lift_ci_low_95": float(np.quantile(arr, 0.025)),
                "lift_ci_high_95": float(np.quantile(arr, 0.975)),
                "share_bootstraps_lift_gt_1": float((arr > 1.0).mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["share_bootstraps_lift_gt_1", "lift_mean"], ascending=[False, False]
    )


def choose_goals(features: pd.DataFrame, drivers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    org_goals = features[["organization_id", "converted"]].copy()

    for goal, activities in GOAL_CANDIDATES.items():
        cols = [f"did__{a}" for a in activities if f"did__{a}" in features.columns]
        if cols:
            org_goals[goal] = (features[cols].sum(axis=1) > 0).astype(int)
        else:
            org_goals[goal] = 0

    goal_cols = [c for c in org_goals.columns if c.startswith("goal_")]
    org_goals["is_activated"] = (org_goals[goal_cols].sum(axis=1) == len(goal_cols)).astype(int)

    summary_rows = []
    for goal in goal_cols + ["is_activated"]:
        with_goal = org_goals[goal] == 1
        conv_with = org_goals.loc[with_goal, "converted"].mean()
        conv_without = org_goals.loc[~with_goal, "converted"].mean()
        summary_rows.append(
            {
                "goal_name": goal,
                "completion_rate": org_goals[goal].mean(),
                "conv_rate_when_completed": conv_with,
                "conv_rate_when_not_completed": conv_without,
                "lift": conv_with / max(conv_without, 1e-9),
            }
        )

    goal_summary = pd.DataFrame(summary_rows).sort_values("lift", ascending=False)

    # Pick the strongest supporting activity per goal.
    goal_evidence_rows = []
    for goal_name in goal_summary["goal_name"]:
        if goal_name == "is_activated":
            goal_evidence_rows.append(
                {
                    "goal_name": goal_name,
                    "activity_evidence__activity_name": pd.NA,
                    "activity_evidence__lift": pd.NA,
                    "activity_evidence__p_value": pd.NA,
                    "activity_evidence__p_value_adj_bh": pd.NA,
                    "activity_evidence__coverage_rate": pd.NA,
                }
            )
            continue

        candidates = GOAL_CANDIDATES.get(goal_name, [])
        candidates_df = drivers.loc[drivers["activity_name"].isin(candidates)].copy()
        if candidates_df.empty:
            goal_evidence_rows.append(
                {
                    "goal_name": goal_name,
                    "activity_evidence__activity_name": pd.NA,
                    "activity_evidence__lift": pd.NA,
                    "activity_evidence__p_value": pd.NA,
                    "activity_evidence__p_value_adj_bh": pd.NA,
                    "activity_evidence__coverage_rate": pd.NA,
                }
            )
            continue

        top_row = candidates_df.sort_values(["p_value_adj_bh", "lift"], ascending=[True, False]).iloc[0]
        goal_evidence_rows.append(
            {
                "goal_name": goal_name,
                "activity_evidence__activity_name": top_row["activity_name"],
                "activity_evidence__lift": top_row["lift"],
                "activity_evidence__p_value": top_row["p_value"],
                "activity_evidence__p_value_adj_bh": top_row["p_value_adj_bh"],
                "activity_evidence__coverage_rate": top_row["coverage_rate"],
            }
        )

    goal_evidence = pd.DataFrame(goal_evidence_rows)
    return org_goals, goal_summary.merge(goal_evidence, on="goal_name", how="left")


def save_outputs(
    df: pd.DataFrame,
    features: pd.DataFrame,
    drivers: pd.DataFrame,
    logit_coef: pd.DataFrame,
    importance: pd.DataFrame,
    segments: pd.DataFrame,
    stability: pd.DataFrame,
    goals_org: pd.DataFrame,
    goals_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    tables_dir = output_dir / "tables"
    figs_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Method A, Fisher exact tests with BH correction.
    drivers.to_csv(tables_dir / "conversion_driver_activity_stats.csv", index=False)

    # Method B, logistic coefficients and odds ratios.
    logit_coef.to_csv(tables_dir / "conversion_driver_logistic_coefficients.csv", index=False)

    # Method C, random forest feature importance.
    importance.to_csv(tables_dir / "conversion_driver_model_coefficients.csv", index=False)

    # Method D, engagement segments by depth and breadth.
    segments.to_csv(tables_dir / "conversion_driver_engagement_segments.csv", index=False)

    # Method E, bootstrap stability for top lifts.
    stability.to_csv(tables_dir / "conversion_driver_bootstrap_stability.csv", index=False)

    goals_org.to_csv(tables_dir / "org_trial_goals_and_activation.csv", index=False)
    goals_summary.to_csv(tables_dir / "goal_summary.csv", index=False)

    quality = pd.DataFrame(
        {
            "metric": [
                "rows_clean",
                "organizations",
                "activities",
                "overall_conversion_rate",
            ],
            "value": [
                len(df),
                df["organization_id"].nunique(),
                df["activity_name"].nunique(),
                features["converted"].mean(),
            ],
        }
    )
    quality.to_csv(tables_dir / "data_quality_summary.csv", index=False)

    sns.set_theme(style="whitegrid")

    top_drivers = drivers.sort_values("lift", ascending=False).head(12).sort_values("lift")
    plt.figure(figsize=(10, 6))
    plt.barh(top_drivers["activity_name"], top_drivers["lift"], color="#1f77b4")
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title("Top Activities by Conversion Lift")
    plt.xlabel("Lift, conversion with activity divided by conversion without activity")
    plt.tight_layout()
    plt.savefig(figs_dir / "top_activity_lift.png", dpi=150)
    plt.close()

    goal_cols = [c for c in goals_org.columns if c.startswith("goal_")] + ["is_activated"]
    completion = goals_org[goal_cols].mean().sort_values()
    plt.figure(figsize=(8, 5))
    plt.barh(completion.index, completion.values, color="#2ca02c")
    plt.title("Trial Goal Completion Rates")
    plt.xlabel("Share of organizations")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(figs_dir / "goal_completion_rates.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    df = load_and_clean(args.input)
    features = build_org_features(df)
    drivers = activity_driver_table(features)
    logit_coef = model_logistic_coefficients(features)
    importance = model_importance(features)
    segments = engagement_segmentation(features)
    stability = bootstrap_lift_stability(features, drivers)
    goals_org, goals_summary = choose_goals(features, drivers)

    save_outputs(
        df,
        features,
        drivers,
        logit_coef,
        importance,
        segments,
        stability,
        goals_org,
        goals_summary,
        args.output_dir,
    )

    activated_conv = goals_org.loc[goals_org["is_activated"] == 1, "converted"].mean()
    non_activated_conv = goals_org.loc[goals_org["is_activated"] == 0, "converted"].mean()
    print(f"conversion rate, activated orgs: {activated_conv:.4f}")
    print(f"conversion rate, non-activated orgs: {non_activated_conv:.4f}")


if __name__ == "__main__":
    main()
