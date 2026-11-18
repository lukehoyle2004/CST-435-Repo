-- 001_init.sql -- FineTuneFleet: the corpus, the experiment store, and the
-- prediction log behind a fine-tuned support-ticket router.
--
-- Apply in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Safe to re-run: it drops and recreates the ftf_ objects only.
--
-- Every object is prefixed `ftf_` so this product can share a Supabase project
-- with the others without touching their tables.
--
-- Privacy invariant: a message sent to the live service is NEVER stored.
-- ftf_predictions keeps its SHA-256, its length and what the model answered.
-- ftf_documents holds the public Banking77 corpus (CC-BY-4.0), not customer data.

drop view     if exists ftf_latency;
drop view     if exists ftf_leaderboard;
drop table    if exists ftf_predictions;
drop table    if exists ftf_experiments;
drop table    if exists ftf_documents;

-- ---------------------------------------------------------------------------
-- ftf_documents: the training corpus with its split label, written once by
-- scripts/prepare_data.py --push. A message's dedup key belongs to one split
-- only (enforced by the loader; see README "Dataset").
-- ---------------------------------------------------------------------------
create table ftf_documents (
    id           bigint generated always as identity primary key,
    split        text    not null check (split in ('train', 'validation', 'test')),
    queue        text    not null check (queue in ('card_services', 'transfers', 'top_ups', 'cash_atm',
                                                   'fees_fx', 'payments_disputes', 'security_fraud',
                                                   'account_identity')),
    intent       text    not null,
    text         text    not null,
    text_sha256  text    not null unique check (text_sha256 ~ '^[0-9a-f]{64}$'),
    n_chars      integer not null check (n_chars >= 1),
    n_tokens     integer not null check (n_tokens >= 1),
    loaded_at    timestamptz not null default now()
);
create index idx_ftf_documents_split_queue on ftf_documents (split, queue);

-- ---------------------------------------------------------------------------
-- ftf_experiments: one row per configuration the sweep ran, plus the zero-shot
-- baseline. Metrics are on the held-out TEST split with 95% bootstrap CIs; the
-- validation macro-F1 is what chose the best configuration. delta_* is the
-- paired bootstrap difference against the zero-shot row (same resamples).
-- ---------------------------------------------------------------------------
create table ftf_experiments (
    id                   bigint generated always as identity primary key,
    run_name             text    not null unique,
    kind                 text    not null check (kind in ('fine_tuned', 'zero_shot')),
    base_checkpoint      text    not null,                 -- repo@sha
    seed                 integer not null,
    hyperparams          jsonb   not null default '{}'::jsonb,
    n_train              integer not null check (n_train >= 0),
    val_macro_f1         double precision not null,
    test_accuracy        double precision not null,
    test_accuracy_lo     double precision not null,
    test_accuracy_hi     double precision not null,
    test_macro_f1        double precision not null,
    test_macro_f1_lo     double precision not null,
    test_macro_f1_hi     double precision not null,
    fraud_recall         double precision not null,       -- recall on security_fraud
    fraud_recall_lo      double precision not null,
    fraud_recall_hi      double precision not null,
    delta_macro_f1       double precision,                -- null on the baseline row
    delta_macro_f1_lo    double precision,
    delta_macro_f1_hi    double precision,
    per_queue            jsonb   not null default '{}'::jsonb,  -- precision/recall/f1/support
    train_runtime_s      double precision not null default 0 check (train_runtime_s >= 0),
    eval_ms_per_message  double precision not null check (eval_ms_per_message >= 0),
    artifact_sha256      text,                            -- weights of the kept checkpoint
    is_deployed          boolean not null default false,
    git_sha              text,
    created_at           timestamptz not null default now(),
    check (test_macro_f1_lo <= test_macro_f1 and test_macro_f1 <= test_macro_f1_hi),
    check ((kind = 'zero_shot') = (delta_macro_f1 is null)),
    check (not (is_deployed and kind = 'zero_shot'))
);
-- At most one deployed fine-tuned checkpoint at a time.
create unique index idx_ftf_experiments_one_deployed on ftf_experiments (is_deployed) where is_deployed;

-- ---------------------------------------------------------------------------
-- ftf_predictions: one row per /predict or /predict_zero_shot call, written
-- BEFORE the API responds, and joined to the experiment row that served it.
-- ---------------------------------------------------------------------------
create table ftf_predictions (
    id               bigint generated always as identity primary key,
    experiment_id    bigint  not null references ftf_experiments (id) on delete restrict,
    endpoint         text    not null check (endpoint in ('predict', 'predict_zero_shot')),
    input_sha256     text    not null check (input_sha256 ~ '^[0-9a-f]{64}$'),
    n_chars          integer not null check (n_chars >= 1),
    n_tokens         integer not null check (n_tokens >= 0),
    predicted_queue  text    not null,
    confidence       double precision not null check (confidence between 0 and 1),
    needs_review     boolean not null,                     -- confidence under the review threshold
    scores           jsonb   not null,                     -- queue -> probability
    latency_ms       double precision not null check (latency_ms >= 0),
    load_ms          double precision not null default 0 check (load_ms >= 0),
    cold_start       boolean not null default false,
    source           text    not null default 'api' check (source in ('api', 'benchmark', 'test')),
    api_version      text,
    served_at        timestamptz not null default now(),
    check (cold_start = (load_ms > 0))
);
create index idx_ftf_predictions_served on ftf_predictions (endpoint, served_at desc);
create index idx_ftf_predictions_experiment on ftf_predictions (experiment_id);

-- ---------------------------------------------------------------------------
-- ftf_leaderboard: the experiments ranked by test macro-F1, with intervals.
-- `distinguishable_from_best` is false when this row's CI overlaps the top
-- row's CI -- the interface uses it to say "not meaningfully different".
-- ---------------------------------------------------------------------------
create view ftf_leaderboard with (security_invoker = true) as
with best as (
    select test_macro_f1_lo as best_lo, test_macro_f1_hi as best_hi, run_name as best_run
    from ftf_experiments order by test_macro_f1 desc limit 1
)
select
    rank() over (order by e.test_macro_f1 desc)                    as rank,
    e.id, e.run_name, e.kind, e.base_checkpoint, e.seed, e.hyperparams, e.n_train,
    e.val_macro_f1,
    e.test_accuracy, e.test_accuracy_lo, e.test_accuracy_hi,
    e.test_macro_f1, e.test_macro_f1_lo, e.test_macro_f1_hi,
    e.fraud_recall, e.fraud_recall_lo, e.fraud_recall_hi,
    e.delta_macro_f1, e.delta_macro_f1_lo, e.delta_macro_f1_hi,
    e.train_runtime_s, e.eval_ms_per_message, e.is_deployed, e.created_at,
    (e.run_name = b.best_run or e.test_macro_f1_hi < b.best_lo)    as distinguishable_from_best
from ftf_experiments e cross join best b
order by e.test_macro_f1 desc;

-- ---------------------------------------------------------------------------
-- ftf_latency: server-side latency per endpoint, warm vs cold. Aggregates only:
-- no hashes and no individual predictions. Test rows never count.
-- ---------------------------------------------------------------------------
create view ftf_latency as
select
    p.endpoint,
    min(e.run_name)                                                                     as run_name,
    count(*)                                                                            as n_calls,
    count(*) filter (where p.cold_start)                                                as n_cold,
    round(avg(p.latency_ms)::numeric, 2)                                                as mean_latency_ms,
    round((percentile_cont(0.5)  within group (order by p.latency_ms))::numeric, 2)     as p50_latency_ms,
    round((percentile_cont(0.95) within group (order by p.latency_ms))::numeric, 2)     as p95_latency_ms,
    round(avg(p.load_ms) filter (where p.cold_start)::numeric, 2)                       as mean_load_ms,
    round(avg(p.n_tokens)::numeric, 1)                                                  as mean_tokens,
    round(avg(case when p.needs_review then 1.0 else 0.0 end)::numeric, 4)              as review_rate,
    max(p.served_at)                                                                    as last_served_at
from ftf_predictions p join ftf_experiments e on e.id = p.experiment_id
where p.source <> 'test'
group by p.endpoint;

-- ---------------------------------------------------------------------------
-- Access control. The API writes with the SERVICE-ROLE key (bypasses RLS).
-- The public anon key (FleetDeck, Product 6B) may read the experiment store
-- and the latency aggregates -- never the corpus text or a prediction row.
-- ---------------------------------------------------------------------------
alter table ftf_documents   enable row level security;
alter table ftf_experiments enable row level security;
alter table ftf_predictions enable row level security;

create policy ftf_experiments_public_read on ftf_experiments
    for select to anon, authenticated using (true);

revoke all on ftf_documents, ftf_predictions from anon, authenticated;
revoke insert, update, delete, truncate on ftf_experiments from anon, authenticated;
grant select on ftf_experiments to anon, authenticated;
grant select on ftf_leaderboard to anon, authenticated;
revoke all on ftf_latency from anon, authenticated;
grant select on ftf_latency to anon, authenticated;
