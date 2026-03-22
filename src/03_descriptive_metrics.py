from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CORE_MODULES = {
    "scheduling": [
        "Scheduling.Availability.Set",
        "Scheduling.Shift.Created",
        "Scheduling.Template.ApplyModal.Applied",
        "Scheduling.Shift.AssignmentChanged",
        "Scheduling.ShiftSwap.Created",
        "Scheduling.ShiftSwap.Accepted",
        "Scheduling.ShiftHandover.Created",
        "Scheduling.ShiftHandover.Accepted",
        "Scheduling.OpenShiftRequest.Created",
        "Scheduling.OpenShiftRequest.Approved",
        "Scheduling.Shift.Approved",
    ],
    "time_tracking": [
        "PunchClock.PunchedIn",
        "PunchClock.PunchedOut",
        "PunchClock.Entry.Edited",
        "Break.Activate.Started",
        "Break.Activate.Finished",
        "PunchClockStartNote.Add.Completed",
        "PunchClockEndNote.Add.Completed",
    ],
    "absence": [
        "Absence.Request.Created",
        "Absence.Request.Approved",
        "Absence.Request.Rejected",
    ],
    "communication": ["Communication.Message.Created"],
    "payroll": [
        "Timesheets.BulkApprove.Confirmed",
        "Integration.Xero.PayrollExport.Synced",
        "Revenue.Budgets.Created",
    ],
    "mobile_usage": ["Mobile.Schedule.Loaded", "Shift.View.Opened", "ShiftDetails.View.Opened"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/da_task.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--goal_table",
        type=Path,
        default=Path("outputs/tables/org_trial_goals_and_activation.csv"),
    )
    return parser.parse_args()


def clean_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ["timestamp", "converted_at", "trial_start", "trial_end"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    df["converted"] = (
        df["converted"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    )
    df = df.dropna(subset=["organization_id", "activity_name", "timestamp", "trial_start", "trial_end", "converted"])

    df = df.drop_duplicates().copy()
    df = df.loc[(df["timestamp"] >= df["trial_start"]) & (df["timestamp"] <= df["trial_end"])].copy()
    df["trial_day"] = (df["timestamp"] - df["trial_start"]).dt.days

    return df


def org_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    org = (
        df.groupby("organization_id", as_index=False)
        .agg(
            converted=("converted", "max"),
            trial_start=("trial_start", "min"),
            converted_at=("converted_at", "max"),
            total_events=("activity_name", "size"),
            active_days=("trial_day", "nunique"),
            unique_activities=("activity_name", "nunique"),
        )
    )

    org["events_per_active_day"] = org["total_events"] / org["active_days"].clip(lower=1)
    org["time_to_convert_days"] = (org["converted_at"] - org["trial_start"]).dt.total_seconds() / (3600 * 24)

    for module, activities in CORE_MODULES.items():
        used = (
            df.loc[df["activity_name"].isin(activities)]
            .groupby("organization_id")
            .size()
            .rename(f"used_{module}")
            .gt(0)
            .astype(int)
        )
        org = org.merge(used, on="organization_id", how="left")
        org[f"used_{module}"] = org[f"used_{module}"].fillna(0).astype(int)

    org["module_breadth"] = org[[c for c in org.columns if c.startswith("used_")]].sum(axis=1)
    return org


def save_tables_and_charts(df: pd.DataFrame, org: pd.DataFrame, args: argparse.Namespace) -> None:
    out_tables = args.output_dir / "tables"
    out_figs = args.output_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        {
            "metric": [
                "organizations",
                "events",
                "conversion_rate",
                "median_time_to_convert_days",
                "median_events_per_org",
                "median_active_days_per_org",
                "median_module_breadth",
            ],
            "value": [
                org["organization_id"].nunique(),
                len(df),
                org["converted"].mean(),
                org.loc[org["converted"] == 1, "time_to_convert_days"].median(),
                org["total_events"].median(),
                org["active_days"].median(),
                org["module_breadth"].median(),
            ],
        }
    )

    module_cols = [c for c in org.columns if c.startswith("used_")]
    module_adoption = (
        org[module_cols + ["converted"]]
        .melt(id_vars="converted", var_name="module", value_name="used")
        .groupby(["module", "converted"], as_index=False)["used"]
        .mean()
    )

    trial_day_retention = (
        df.groupby(["organization_id", "trial_day"], as_index=False)
        .size()
        .assign(active=1)
        .groupby("trial_day", as_index=False)["active"]
        .mean()
        .sort_values("trial_day")
    )

    conversion_by_breadth = (
        org.groupby("module_breadth", as_index=False)["converted"]
        .mean()
        .sort_values("module_breadth")
    )

    activity_volume = (
        df.groupby("activity_name", as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
        .sort_values("event_count", ascending=False)
    )

    summary.to_csv(out_tables / "descriptive_summary_metrics.csv", index=False)
    module_adoption.to_csv(out_tables / "module_adoption_by_conversion.csv", index=False)
    trial_day_retention.to_csv(out_tables / "trial_day_activity_rate.csv", index=False)
    conversion_by_breadth.to_csv(out_tables / "conversion_by_module_breadth.csv", index=False)
    activity_volume.to_csv(out_tables / "activity_volume.csv", index=False)
    org.to_csv(out_tables / "org_level_metrics.csv", index=False)

    if args.goal_table.exists():
        goals = pd.read_csv(args.goal_table)
        if "is_activated" in goals.columns:
            activation_conv = (
                goals.groupby("is_activated", as_index=False)["converted"].mean()
                .rename(columns={"converted": "conversion_rate"})
            )
            activation_conv.to_csv(out_tables / "conversion_by_activation.csv", index=False)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    top_act = activity_volume.head(12).sort_values("event_count")
    plt.barh(top_act["activity_name"], top_act["event_count"], color="#4c78a8")
    plt.title("Top 12 Activities by Event Volume")
    plt.xlabel("Event Count")
    plt.tight_layout()
    plt.savefig(out_figs / "top_activities_by_volume.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=trial_day_retention, x="trial_day", y="active", marker="o", color="#f58518")
    plt.title("Trial Engagement Curve (Share Active by Trial Day)")
    plt.xlabel("Trial Day")
    plt.ylabel("Share of organizations active")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_figs / "trial_day_engagement_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=conversion_by_breadth,
        x="module_breadth",
        y="converted",
        marker="o",
        color="#54a24b",
    )
    plt.title("Conversion Rate by Module Breadth")
    plt.xlabel("Number of modules adopted")
    plt.ylabel("Conversion rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_figs / "conversion_by_module_breadth.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    df = clean_events(args.input)
    org = org_level_metrics(df)
    save_tables_and_charts(df, org, args)

    print(f"organizations={org['organization_id'].nunique()}")
    print(f"conversion_rate={org['converted'].mean():.4f}")
    print(f"median_time_to_convert_days={org.loc[org['converted']==1, 'time_to_convert_days'].median():.2f}")


if __name__ == "__main__":
    main()
