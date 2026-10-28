-- 001_init.sql -- Seq-Sight: IMDB sentiment with a plain LSTM vs an LSTM + attention.
-- Apply in the Supabase dashboard: SQL Editor -> New query -> paste -> Run,
-- then `python -m db.seed_runs` to load models/runs.json.
--
-- Every object is prefixed `seq_` so this can share a Supabase project with the
-- other products (free tier = 2 projects) without touching their `runs` /
-- `predictions` tables. Safe to re-run: it drops and recreates the seq_ objects only.
--
-- Privacy invariant: the review text is NEVER stored. A prediction is reduced to
-- the text's SHA-256 hash, its token counts, the model's answer, and the
-- attention vector (one weight per kept token -- numbers, not words).

drop view  if exists seq_model_comparison;
drop table if exists seq_predictions;
drop table if exists seq_runs;

-- ---------------------------------------------------------------------------
-- seq_runs: one row per offline training run (scripts/train.py + scripts/analyze.py).
-- Both runs share seed, vocab, data order, epochs, batch size, lr and hardware.
-- ---------------------------------------------------------------------------
create table seq_runs (
    run_id               text primary key,              -- 8-hex id from train.py
    name                 text not null unique,          -- 'lstm' | 'attention'
    display_name         text not null,
    architecture         text not null,
    dataset              text not null,
    epochs               integer not null,
    batch_size           integer not null,
    lr                   double precision not null,
    seed                 integer not null,
    params               integer not null,
    max_len              integer not null,
    train_reviews        integer not null,
    val_reviews          integer not null,
    test_reviews         integer not null,
    hardware             text,
    train_seconds        double precision,
    best_epoch           integer not null,
    val_acc              double precision not null,     -- fraction, at best_epoch
    vocab_sha256         text not null,
    test_acc             double precision not null,     -- fraction, all 25k test reviews
    test_acc_ci95        jsonb not null,                -- [low, high] Wilson 95% interval
    macro_f1             double precision not null,
    per_class            jsonb not null,                -- {class: {precision, recall, f1}}
    ece                  double precision not null,     -- expected calibration error
    confident_errors     integer not null,              -- wrong with confidence >= 0.9
    confident_error_rate double precision not null,     -- share of all errors
    selective            jsonb not null,                -- coverage/accuracy if it abstains below a confidence
    length_buckets       jsonb not null,                -- [{bucket, n, acc}] by raw length
    history              jsonb not null,                -- per-epoch train/val loss, val acc
    attention_stats      jsonb,                         -- attention run only
    agreement            jsonb not null,                -- both right / only one right / both wrong + McNemar p
    created_at           timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- seq_predictions: one row per /predict call, with the attention vector the
-- Recent Predictions drilldown draws as a heatmap.
-- ---------------------------------------------------------------------------
create table seq_predictions (
    id                bigint generated always as identity primary key,
    text_sha256       text not null check (text_sha256 ~ '^[0-9a-f]{64}$'),
    source            text not null check (source in ('user', 'imdb_test')),
    test_index        integer,                       -- public IMDB test review, else null
    n_tokens          integer not null check (n_tokens between 1 and 256),  -- tokens scored
    raw_tokens        integer not null check (raw_tokens >= n_tokens),      -- before truncation
    unk_tokens        integer not null check (unk_tokens >= 0),
    predicted_label   integer not null check (predicted_label in (0, 1)),
    predicted_class   text not null,
    confidence        double precision not null check (confidence between 0 and 1),
    probabilities     jsonb not null,                -- {negative: p, positive: p}
    abstained         boolean not null default false,  -- guardrail: too uncertain to call
    flags             jsonb not null default '[]',     -- guardrail codes, e.g. ["low_confidence"]
    model             text not null check (model in ('lstm', 'attention')),
    attention         jsonb check (attention is null or jsonb_typeof(attention) = 'array'),
    served_by_run_id  text not null references seq_runs (run_id) on delete cascade,
    latency_ms        double precision,
    created_at        timestamptz not null default now(),
    check ((source = 'imdb_test') = (test_index is not null)),
    check ((model = 'attention') = (attention is not null)),
    check (attention is null or jsonb_array_length(attention) = n_tokens)
);

create index idx_seq_predictions_created_at on seq_predictions (created_at desc);
create index idx_seq_predictions_run        on seq_predictions (served_by_run_id);
create index idx_seq_predictions_sha        on seq_predictions (text_sha256);

-- ---------------------------------------------------------------------------
-- seq_model_comparison: the headline LSTM-vs-attention numbers, straight from SQL.
-- ---------------------------------------------------------------------------
create view seq_model_comparison with (security_invoker = true) as
select r.name,
       r.display_name,
       r.run_id,
       r.seed,
       r.epochs,
       r.best_epoch,
       r.params,
       round((100 * r.test_acc)::numeric, 2)                     as test_acc_pct,
       round(r.macro_f1::numeric, 4)                             as macro_f1,
       round(r.ece::numeric, 4)                                  as ece,
       r.confident_errors,
       round((100 * (r.test_acc - l.test_acc))::numeric, 2)      as gain_over_lstm_pts,
       (select count(*) from seq_predictions p where p.served_by_run_id = r.run_id) as served_predictions
from seq_runs r
cross join (select test_acc from seq_runs where name = 'lstm') l;

-- ---------------------------------------------------------------------------
-- Row Level Security. The API writes with the SERVICE-ROLE key (bypasses RLS).
-- The Streamlit UI reads with the public ANON key: SELECT only, no writes.
-- seq_predictions holds hashes, counts, labels and attention numbers -- no
-- text -- so it is safe to show on the public Recent Predictions tab.
-- ---------------------------------------------------------------------------
alter table seq_runs        enable row level security;
alter table seq_predictions enable row level security;

create policy "anon read seq_runs"        on seq_runs        for select to anon using (true);
create policy "anon read seq_predictions" on seq_predictions for select to anon using (true);

grant select on seq_model_comparison to anon;
