-- 002_benchmarks_hourly.sql -- latency per model OVER TIME, for the Product 5B
-- "Latency History" chart. Run after 001_init.sql (SQL Editor -> paste -> Run).
-- Safe to re-run.
--
-- trio_benchmarks (001) aggregates all calls per model. This view adds the time
-- axis: one row per model per hour. Like 001's view it excludes source='test'
-- rows and exposes only aggregates, so the public anon key may read it.

drop materialized view if exists trio_benchmarks_hourly;

create materialized view trio_benchmarks_hourly as
select
    i.model,
    date_trunc('hour', i.served_at)                                                   as hour,
    count(*)                                                                          as n_calls,
    count(*) filter (where i.cold_start)                                              as n_cold,
    round((percentile_cont(0.5)  within group (order by i.latency_ms))::numeric, 2)   as p50_latency_ms,
    round((percentile_cont(0.95) within group (order by i.latency_ms))::numeric, 2)   as p95_latency_ms,
    round((percentile_cont(0.5)  within group (order by i.total_ms))::numeric, 2)     as p50_total_ms,
    round(avg(i.load_ms) filter (where i.cold_start)::numeric, 2)                     as mean_load_ms,
    now()                                                                             as refreshed_at
from trio_inferences i
where i.source <> 'test'
group by i.model, date_trunc('hour', i.served_at);

create unique index idx_trio_benchmarks_hourly on trio_benchmarks_hourly (model, hour);

-- One refresh call (GET /benchmarks) now refreshes both views.
create or replace function trio_refresh_benchmarks() returns timestamptz
language plpgsql
security definer
set search_path = public
as $$
begin
    refresh materialized view concurrently trio_benchmarks;
    refresh materialized view concurrently trio_benchmarks_hourly;
    return now();
end;
$$;

revoke all on trio_benchmarks_hourly from anon, authenticated;
grant select on trio_benchmarks_hourly to anon, authenticated;
revoke execute on function trio_refresh_benchmarks() from public, anon, authenticated;
grant  execute on function trio_refresh_benchmarks() to service_role;
