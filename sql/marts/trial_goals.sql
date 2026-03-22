-- Mart model at organization grain with one boolean flag per trial goal.
-- Goal hypotheses are based on conversion-driver analysis and product-value logic.

with events as (
    select *
    from {{ ref('stg_trial_events') }}
),

org_activity as (
    select
        organization_id,
        max(converted) as converted,
        max(case when activity_name in (
            'Scheduling.Shift.Created',
            'Scheduling.Template.ApplyModal.Applied'
        ) then 1 else 0 end) as goal_schedule_setup,
        max(case when activity_name in (
            'Mobile.Schedule.Loaded',
            'Shift.View.Opened',
            'ShiftDetails.View.Opened'
        ) then 1 else 0 end) as goal_schedule_visibility,
        max(case when activity_name in (
            'PunchClock.PunchedIn',
            'PunchClock.PunchedOut'
        ) then 1 else 0 end) as goal_time_tracking_started,
        max(case when activity_name in (
            'Scheduling.Shift.AssignmentChanged',
            'Scheduling.Shift.Approved',
            'Scheduling.OpenShiftRequest.Created'
        ) then 1 else 0 end) as goal_ops_approval
    from events
    group by organization_id
)

select
    organization_id,
    converted,
    goal_schedule_setup,
    goal_schedule_visibility,
    goal_time_tracking_started,
    goal_ops_approval
from org_activity;
