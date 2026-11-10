-- 001_init.sql -- TrioFormer: one service, four transformers (hand-written
-- encoder block, BERT, GPT-2, T5), and the latency data behind the cost story.
--
-- Apply in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Safe to re-run: it drops and recreates the trio_ objects only.
--
-- Every object is prefixed `trio_` so the project could share a Supabase
-- project with the other products without touching their tables.
--
-- Privacy invariant: a submitted prompt is NEVER stored. trio_inputs keeps only
-- its SHA-256 hash and its size. trio_inferences keeps what the model produced
-- (required to audit the service) plus the server-side latency. Only the
-- aggregate trio_benchmarks view is readable with the public anon key.

drop function          if exists trio_log_inference(text, integer, integer, text, text, text, text, jsonb, double precision, double precision, boolean, integer, integer, text, text);
drop function          if exists trio_refresh_benchmarks();
drop materialized view if exists trio_benchmarks;
drop table             if exists trio_inferences;
drop table             if exists trio_inputs;

-- ---------------------------------------------------------------------------
-- trio_inputs: one row per distinct submitted prompt, identified by its hash.
-- Submitting the same paragraph to all four endpoints (the Compare view)
-- produces ONE input row and four inference rows that share it.
-- ---------------------------------------------------------------------------
create table trio_inputs (
    id              bigint generated always as identity primary key,
    input_sha256    text        not null unique check (input_sha256 ~ '^[0-9a-f]{64}$'),
    n_chars         integer     not null check (n_chars >= 1),
    n_words         integer     not null check (n_words >= 0),
    times_submitted integer     not null default 1 check (times_submitted >= 1),
    first_seen_at   timestamptz not null default now(),
    last_seen_at    timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- trio_inferences: one row per model call, written BEFORE the API responds.
--   latency_ms   wall time of the model computation alone (tokenise -> forward
--                / generate -> decode), measured on the server
--   load_ms      time spent loading the checkpoint into memory during this
--                request (0 when it was already resident) -- the cold-start cost
--   total_ms     latency_ms + load_ms + request handling, i.e. what a caller paid
-- ---------------------------------------------------------------------------
create table trio_inferences (
    id              bigint generated always as identity primary key,
    input_id        bigint      not null references trio_inputs (id) on delete cascade,
    model           text        not null check (model in ('encoder_block', 'bert', 'gpt2', 't5')),
    archetype       text        not null check (archetype in ('encoder', 'decoder', 'encoder-decoder')),
    checkpoint      text        not null,              -- e.g. distilgpt2@<commit>
    task            text        not null check (task in ('attention', 'fill-mask', 'continue', 'summarize')),
    output          jsonb       not null,              -- the model's answer (see README privacy notes)
    latency_ms      double precision not null check (latency_ms >= 0),
    load_ms         double precision not null default 0 check (load_ms >= 0),
    total_ms        double precision not null check (total_ms >= latency_ms),
    cold_start      boolean     not null default false,
    input_tokens    integer     not null check (input_tokens >= 0),
    output_tokens   integer     not null check (output_tokens >= 0),
    source          text        not null default 'api' check (source in ('api', 'benchmark', 'test')),
    api_version     text,                              -- git sha that served it
    served_at       timestamptz not null default now(),
    check ((model = 'encoder_block') = (task = 'attention')),
    check (cold_start = (load_ms > 0))
);

create index idx_trio_inferences_model_served on trio_inferences (model, served_at desc);
create index idx_trio_inferences_input        on trio_inferences (input_id);

-- ---------------------------------------------------------------------------
-- trio_benchmarks: latency aggregated by model. Warm calls (model already in
-- memory) and cold starts are split, because on a 512 MB free instance that
-- swaps models in and out, the two tell very different cost stories.
-- ---------------------------------------------------------------------------
create materialized view trio_benchmarks as
select
    i.model,
    min(i.archetype)                                                                  as archetype,
    min(i.checkpoint)                                                                 as checkpoint,
    count(*)                                                                          as n_calls,
    count(*) filter (where not i.cold_start)                                          as n_warm,
    count(*) filter (where i.cold_start)                                              as n_cold,
    round(avg(i.latency_ms)::numeric, 2)                                              as mean_latency_ms,
    round((percentile_cont(0.5)  within group (order by i.latency_ms))::numeric, 2)   as p50_latency_ms,
    round((percentile_cont(0.95) within group (order by i.latency_ms))::numeric, 2)   as p95_latency_ms,
    round(min(i.latency_ms)::numeric, 2)                                              as min_latency_ms,
    round(max(i.latency_ms)::numeric, 2)                                              as max_latency_ms,
    round(avg(i.load_ms) filter (where i.cold_start)::numeric, 2)                     as mean_load_ms,
    round(avg(i.total_ms)::numeric, 2)                                                as mean_total_ms,
    round(avg(i.input_tokens)::numeric, 1)                                            as mean_input_tokens,
    round(avg(i.output_tokens)::numeric, 1)                                           as mean_output_tokens,
    round((sum(i.latency_ms) / nullif(sum(i.output_tokens), 0))::numeric, 2)          as ms_per_output_token,
    count(distinct i.input_id)                                                        as distinct_inputs,
    min(i.served_at)                                                                  as first_served_at,
    max(i.served_at)                                                                  as last_served_at,
    now()                                                                             as refreshed_at
from trio_inferences i
where i.source <> 'test'              -- live-test rows never reach the published numbers
group by i.model;

-- A unique index lets the view refresh CONCURRENTLY (readers never block).
create unique index idx_trio_benchmarks_model on trio_benchmarks (model);

-- ---------------------------------------------------------------------------
-- trio_log_inference: upsert the input by hash and insert the inference in
-- one transaction (one round trip from the API). Returns the inference id.
-- ---------------------------------------------------------------------------
create function trio_log_inference(
    p_input_sha256  text,
    p_n_chars       integer,
    p_n_words       integer,
    p_model         text,
    p_archetype     text,
    p_checkpoint    text,
    p_task          text,
    p_output        jsonb,
    p_latency_ms    double precision,
    p_load_ms       double precision,
    p_cold_start    boolean,
    p_input_tokens  integer,
    p_output_tokens integer,
    p_source        text,
    p_api_version   text
) returns bigint
language plpgsql
security invoker
set search_path = public
as $$
declare
    v_input_id     bigint;
    v_inference_id bigint;
begin
    insert into trio_inputs (input_sha256, n_chars, n_words)
    values (p_input_sha256, p_n_chars, p_n_words)
    on conflict (input_sha256) do update
        set times_submitted = trio_inputs.times_submitted + 1,
            last_seen_at    = now()
    returning id into v_input_id;

    insert into trio_inferences (input_id, model, archetype, checkpoint, task, output,
                                 latency_ms, load_ms, total_ms, cold_start,
                                 input_tokens, output_tokens, source, api_version)
    values (v_input_id, p_model, p_archetype, p_checkpoint, p_task, p_output,
            p_latency_ms, p_load_ms, p_latency_ms + p_load_ms, p_cold_start,
            p_input_tokens, p_output_tokens, p_source,
            p_api_version)
    returning id into v_inference_id;

    return v_inference_id;
end;
$$;

-- ---------------------------------------------------------------------------
-- trio_refresh_benchmarks: called by GET /benchmarks (throttled) and by
-- scripts/benchmark_latency.py. Runs with the owner's rights so the
-- service role can refresh a view it does not own.
-- ---------------------------------------------------------------------------
create function trio_refresh_benchmarks() returns timestamptz
language plpgsql
security definer
set search_path = public
as $$
begin
    refresh materialized view concurrently trio_benchmarks;
    return now();
end;
$$;

-- ---------------------------------------------------------------------------
-- Access control. The API writes with the SERVICE-ROLE key (bypasses RLS).
-- The public anon key (Product 5B's Streamlit UI) may read ONLY the aggregate
-- benchmarks view: not the hashes, and not the model outputs.
-- ---------------------------------------------------------------------------
alter table trio_inputs     enable row level security;
alter table trio_inferences enable row level security;
-- (no anon policies: RLS on with no policy = no rows for anon)

revoke all on trio_benchmarks from anon, authenticated;
grant select on trio_benchmarks to anon, authenticated;

revoke execute on function trio_log_inference(text, integer, integer, text, text, text, text, jsonb, double precision, double precision, boolean, integer, integer, text, text) from public, anon, authenticated;
revoke execute on function trio_refresh_benchmarks() from public, anon, authenticated;
grant  execute on function trio_log_inference(text, integer, integer, text, text, text, text, jsonb, double precision, double precision, boolean, integer, integer, text, text) to service_role;
grant  execute on function trio_refresh_benchmarks() to service_role;
