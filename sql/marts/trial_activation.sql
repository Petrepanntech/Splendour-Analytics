-- Mart model at organization grain: full trial activation status.

with goals as (
    select *
    from {{ ref('trial_goals') }}
)

select
    organization_id,
    converted,
    goal_schedule_setup,
    goal_schedule_visibility,
    goal_time_tracking_started,
    goal_ops_approval,
    case
        when goal_schedule_setup = 1
         and goal_schedule_visibility = 1
         and goal_time_tracking_started = 1
         and goal_ops_approval = 1
        then 1
        else 0
    end as is_activated
from goals;
