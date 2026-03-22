# Trial Activation Analysis, Splendor Analytics Challenge

I built this repo as my full solution to the Trial Activation community challenge.

## Objective
I define measurable trial activation goals for trial organizations, build marts-layer SQL models to track goals and activation, and produce descriptive analytics for product decisions.

## Repository Structure

- `src/01_explore_and_define_goals.py`: Task 1 analysis, cleaning, EDA, conversion drivers, trial goal definition.
- `src/03_descriptive_metrics.py`: Task 3 descriptive product analytics.
- `sql/staging/stg_trial_events.sql`: staging model for cleaned event data.
- `sql/marts/trial_goals.sql`: mart model with one row per organization and one column per goal.
- `sql/marts/trial_activation.sql`: mart model with activation status.
- `outputs/tables/`: analysis outputs and KPI tables.
- `outputs/figures/`: visual outputs.
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

## Task 1, Approach

1. Clean and normalize event data
- Standardize column names.
- Parse datetime fields.
- Drop exact duplicate events.
- Keep only events inside the true trial window, `trial_start` to `trial_end`.
- Create derived fields, `trial_day` and `event_date`.

2. Build an organization-level feature table
- Event volume, active days, feature breadth.
- Activity-level counts and binary adoption flags.

3. Run conversion-driver analysis with multiple methods
- Method A: activity-level conversion lift plus Fisher exact tests.
- Apply Benjamini-Hochberg correction, `p_value_adj_bh`, to control false discoveries across many activity tests.
- Method B: supervised model, Random Forest, to rank feature importance.

4. Define trial goals, hypothesis-driven and grounded in product value
- `goal_schedule_setup`: shift created OR template applied.
- `goal_schedule_visibility`: schedule and shift views.
- `goal_time_tracking_started`: punch-in OR punch-out.
- `goal_ops_approval`: assignment change OR shift approval OR open-shift request creation.

5. Define trial activation
- An organization is activated only when all four goals are complete.

## Task 1, Key Findings

From this dataset run:
- Organizations: 966
- Clean in-trial events: 102,895, from 170,526 raw, with many exact duplicates removed
- Overall conversion rate: 21.33%

Important analytical result:
- Activity-level lifts are mostly small and often statistically weak.
- The predictive model overfits and generalizes poorly, `train_auc=0.852`, `test_auc=0.467`, so current event telemetry alone does not predict conversion well.

Interpretation:
- I treat trial goals as activation hypotheses and product-value milestones, not guaranteed causal levers.

## Task 2, SQL Models, Marts

Two marts are included:

1. `trial_goals`
- Grain: one row per `organization_id`
- Tracks completion of each goal flag

2. `trial_activation`
- Grain: one row per `organization_id`
- Adds `is_activated` when all goals are complete

Both models depend on `stg_trial_events` for deduplicated, in-window events.

## Task 3, Descriptive Analytics

Generated in `outputs/tables/` and `outputs/figures/`:

- Core KPIs: conversion rate, median time to convert, event depth, active-day depth.
- Module adoption by conversion segment.
- Trial-day engagement curve.
- Engagement curve is computed as active organizations on day $d$ divided by total organizations in the cohort.
- Conversion by module breadth.
- Activity volume ranking.
- Conversion by activation status.

Selected metrics from this run:
- Median time to convert: about 30.02 days
- Median events per organization: 8
- Median active days per organization: 1
- Median module breadth: 1

## Product Recommendations

1. Improve event instrumentation quality
- Add richer onboarding events and user-role context to increase predictive signal.

2. Focus onboarding on breadth, not only depth
- Most organizations show low active-day and module breadth during trial.

3. Use activation goals as operating health milestones
- Build interventions around incomplete goal paths, then track outcomes before treating them as causal drivers.

4. Add experiments to test activation hypotheses
- Run A/B tests on onboarding interventions tied to each goal and measure conversion uplift.

## Notes

- SQL files use dbt-style `source()` and `ref()` macros. If you are not using dbt, map object names to your warehouse.
- Date-diff functions may need small dialect changes based on your SQL engine.
