-- Staging model at event grain: one row per in-trial product event.
-- Replace {{ source('raw', 'trial_events') }} with your raw table reference if not using dbt.

with source_events as (
    select
        cast(organization_id as varchar) as organization_id,
        cast(activity_name as varchar) as activity_name,
        cast("timestamp" as timestamp) as event_ts,
        lower(cast(converted as varchar)) as converted_raw,
        cast(converted_at as timestamp) as converted_at,
        cast(trial_start as timestamp) as trial_start,
        cast(trial_end as timestamp) as trial_end
    from {{ source('raw', 'trial_events') }}
),

deduped as (
    select distinct
        organization_id,
        activity_name,
        event_ts,
        converted_raw,
        converted_at,
        trial_start,
        trial_end
    from source_events
),

cleaned as (
    select
        organization_id,
        activity_name,
        event_ts,
        case
            when converted_raw = 'true' then 1
            when converted_raw = 'false' then 0
            else null
        end as converted,
        converted_at,
        trial_start,
        trial_end,
        cast(date(event_ts) as date) as event_date,
        cast(date_diff('day', trial_start, event_ts) as integer) as trial_day
    from deduped
    where event_ts is not null
      and organization_id is not null
      and activity_name is not null
      and trial_start is not null
      and trial_end is not null
      and event_ts between trial_start and trial_end
)

select *
from cleaned
where converted is not null;
