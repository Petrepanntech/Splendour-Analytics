# Trial Activation Analysis - Splendor Analytics Challenge

This repository contains a full solution to the Trial Activation community challenge.

## Objective
Define measurable Trial Activation goals for trial organizations, build marts-layer SQL models for tracking goals and activation, and provide descriptive analytics for product decision-making.

## Repository Structure

- `src/01_explore_and_define_goals.py`: Task 1 analysis (cleaning, EDA, conversion drivers, trial goal definition).
- `src/03_descriptive_metrics.py`: Task 3 descriptive product analytics.
- `sql/staging/stg_trial_events.sql`: staging model for cleaned event data.
- `sql/marts/trial_goals.sql`: mart model with one row per organization and one column per goal.
- `sql/marts/trial_activation.sql`: mart model with activation status.
- `outputs/tables/`: rendered analysis outputs and KPI tables.
- `outputs/figures/`: rendered visual outputs.
- `data/raw/da_task.csv`: input dataset used in this run.

## Environment

Python: 3.13+

Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

Run Task 1:

```bash
python src/01_explore_and_define_goals.py --input data/raw/da_task.csv --output_dir outputs
```

Run Task 3:

```bash
python src/03_descriptive_metrics.py --input data/raw/da_task.csv --output_dir outputs --goal_table outputs/tables/org_trial_goals_and_activation.csv
```

## Task 1 - Approach

1. Clean and normalize event data
- Standardized column names.
- Parsed datetime fields.
- Removed exact duplicate events.
- Filtered events to the true trial window (`trial_start` to `trial_end`).
- Created derived fields (`trial_day`, `event_date`).

2. Build organization-level feature table
- Event volume, active days, feature breadth.
- Activity-level counts and binary adoption flags.

3. Conversion-driver analysis using multiple methods
- Method A: Activity-level conversion lift + Fisher exact tests.
- Applied Benjamini-Hochberg correction (`p_value_adj_bh`) to control false discoveries across many activity tests.
- Method B: Supervised model (Random Forest) to rank feature importance.

4. Define Trial Goals (hypothesis-driven, product-value grounded)
- `goal_schedule_setup`: shift created OR template applied.
- `goal_schedule_visibility`: schedule/shift views.
- `goal_time_tracking_started`: punch-in OR punch-out.
- `goal_ops_approval`: assignment change OR shift approval OR open-shift request creation.

5. Define Trial Activation
- An organization is activated only if all four goals are completed.

## Task 1 - Key Findings

From this dataset run:
- Organizations: 966
- Clean events in-trial: 102,895 (from 170,526 raw; substantial exact duplicates removed)
- Overall conversion rate: 21.33%

Important analytical result:
- Activity-level lifts were mostly small and often statistically weak.
- The predictive model showed overfit and weak generalization (`train_auc=0.852`, `test_auc=0.467`), suggesting current event telemetry alone does not reliably predict conversion.

Interpretation:
- Trial goals are best treated as activation hypotheses and product-value milestones, not guaranteed causal levers.

## Task 2 - SQL Models (Marts)

Two marts are provided:

1. `trial_goals`
- Grain: one row per `organization_id`
- Tracks completion of each goal flag

2. `trial_activation`
- Grain: one row per `organization_id`
- Adds `is_activated` where all goals are complete

Both models depend on `stg_trial_events` for deduplicated, in-window events.

## Task 3 - Descriptive Analytics

Generated in `outputs/tables/` and `outputs/figures/`:

- Core KPIs: conversion rate, median time to convert, event depth, active day depth.
- Module adoption by conversion segment.
- Trial-day engagement curve.
- Engagement curve is computed as: active organizations on day $d$ / total organizations in cohort.
- Conversion by module breadth.
- Activity volume ranking.
- Conversion by activation status.

Selected metrics from this run:
- Median time to convert: ~30.02 days
- Median events per organization: 8
- Median active days per organization: 1
- Median module breadth: 1

## Product Recommendations

1. Improve event instrumentation quality
- Add richer onboarding events and user-role context to increase predictive signal.

2. Focus onboarding on breadth, not only depth
- Most organizations show low active-day and module breadth during trial.

3. Use activation goals operationally as health milestones
- Build interventions around incomplete goal paths, but monitor outcomes before treating as causal drivers.

4. Add experiments to validate activation hypotheses
- A/B test onboarding interventions tied to each goal and measure uplift in conversion.

## Notes

- SQL files use dbt-style `source()` and `ref()` macros; adapt object names for your warehouse if not using dbt.
- Date-diff functions may need minor dialect adjustments depending on your SQL engine.
