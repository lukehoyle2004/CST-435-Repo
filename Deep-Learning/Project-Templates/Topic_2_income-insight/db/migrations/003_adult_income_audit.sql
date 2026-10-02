-- 003_adult_income_audit.sql
-- Moves Income-Insight from synthetic data to the real UCI Adult dataset and
-- adds everything the bias audit, activation comparison, and Model Performance
-- tab read. Apply AFTER 002 in the Supabase SQL Editor (New query -> paste ->
-- Run), then load the data with:  python -m db.load_adult
-- It is idempotent: safe to run more than once.

-- ---------------------------------------------------------------------------
-- adult_income: one row per census record, with a FIXED stratified split so
-- every run trains, validates, and tests on the same rows (controlled
-- comparisons). Protected attributes (sex, race) are stored for auditing.
-- No names or direct identifiers exist in this dataset.
-- ---------------------------------------------------------------------------
create table if not exists adult_income (
    id              bigint primary key,           -- stable row id from the loader
    age             integer not null,
    education_num   integer not null,
    capital_gain    integer not null,
    capital_loss    integer not null,
    hours_per_week  integer not null,
    workclass       text    not null,
    marital_status  text    not null,
    occupation      text    not null,
    relationship    text    not null,
    native_country  text    not null,
    sex             text    not null,
    race            text    not null,
    income          text    not null check (income in ('<=50K', '>50K')),
    label           smallint not null check (label in (0, 1)),
    split           text    not null check (split in ('train', 'val', 'test'))
);
create index if not exists idx_adult_income_split on adult_income (split);

-- Not anon-readable: only the API (service role) reads the raw records.
alter table adult_income enable row level security;

-- ---------------------------------------------------------------------------
-- runs: experiment settings and the full evaluation of each run.
-- ---------------------------------------------------------------------------
-- Runs no longer come from the synthetic `datasets` table.
alter table runs alter column dataset_id drop not null;
alter table runs add column if not exists data_source text not null default 'synthetic';

alter table runs add column if not exists activation        text    not null default 'relu';
alter table runs add column if not exists seed              integer not null default 0;
alter table runs add column if not exists max_train_rows    integer;
alter table runs add column if not exists include_protected boolean not null default false;
alter table runs add column if not exists n_train integer;
alter table runs add column if not exists n_val   integer;
alter table runs add column if not exists n_test  integer;

-- Headline metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc` (from
-- 001) are measured on the TEST split; these add the train/val accuracies.
alter table runs add column if not exists train_accuracy double precision;
alter table runs add column if not exists val_accuracy   double precision;

-- Calibration: Brier score, expected calibration error, reliability bins.
alter table runs add column if not exists brier       double precision;
alter table runs add column if not exists ece         double precision;
alter table runs add column if not exists calibration jsonb;

-- {"tn","fp","fn","tp"} on the test split, per-class P/R/F1, permutation
-- importance (AUC drop per shuffled column), and the validation-loss curve.
alter table runs add column if not exists confusion        jsonb;
alter table runs add column if not exists class_report     jsonb;
alter table runs add column if not exists perm_importance  jsonb;
alter table runs add column if not exists val_loss_history jsonb not null default '[]';

alter table runs drop constraint if exists runs_activation_check;
alter table runs add constraint runs_activation_check
    check (activation in ('relu', 'leaky_relu', 'tanh', 'sigmoid', 'gelu'));

-- ---------------------------------------------------------------------------
-- predictions: link scored test rows back to adult_income (for the bias audit)
-- and record where each prediction came from.
-- ---------------------------------------------------------------------------
alter table predictions add column if not exists adult_id bigint
    references adult_income (id) on delete set null;
alter table predictions add column if not exists source text not null default 'row';
alter table predictions drop constraint if exists predictions_source_check;
alter table predictions add constraint predictions_source_check
    check (source in ('row', 'csv', 'test_eval'));
-- Test-split predictions reference adult_income instead of copying features.
alter table predictions alter column features drop not null;
create index if not exists idx_predictions_adult_id on predictions (adult_id);

-- ---------------------------------------------------------------------------
-- bias_audit: confusion counts, FPR, and FNR per run x protected attribute x
-- group, computed in SQL by joining each run's test-split predictions to the
-- true labels and protected attributes in adult_income.
--   FPR = FP / (FP + TN): share of truly <=50K people wrongly predicted >50K
--   FNR = FN / (FN + TP): share of truly  >50K people the model missed
-- security_invoker makes the view obey the caller's RLS, so the anon key
-- cannot read it; the API (service role) can.
-- ---------------------------------------------------------------------------
create or replace view bias_audit with (security_invoker = true) as
with scored as (
    select p.run_id, a.sex, a.race, a.label as y_true, p.label as y_pred
    from predictions p
    join adult_income a on a.id = p.adult_id
    where p.source = 'test_eval'
),
long as (
    select run_id, 'sex'  as attribute, sex  as grp, y_true, y_pred from scored
    union all
    select run_id, 'race' as attribute, race as grp, y_true, y_pred from scored
)
select
    run_id,
    attribute,
    grp,
    count(*)                                            as n,
    count(*) filter (where y_true = 1 and y_pred = 1)   as tp,
    count(*) filter (where y_true = 0 and y_pred = 1)   as fp,
    count(*) filter (where y_true = 0 and y_pred = 0)   as tn,
    count(*) filter (where y_true = 1 and y_pred = 0)   as fn,
    avg(y_pred::double precision)                       as positive_rate,
    avg(y_true::double precision)                       as base_rate,
    (count(*) filter (where y_true = 0 and y_pred = 1))::double precision
        / nullif(count(*) filter (where y_true = 0), 0) as fpr,
    (count(*) filter (where y_true = 1 and y_pred = 0))::double precision
        / nullif(count(*) filter (where y_true = 1), 0) as fnr
from long
group by run_id, attribute, grp;

-- ---------------------------------------------------------------------------
-- activation_comparison: SQL summary of ok runs on the real data, grouped by
-- activation and the matched controls (depth, width, lr, batch, epochs, seed,
-- training budget). Rows sharing every control differ ONLY in activation.
-- ---------------------------------------------------------------------------
create or replace view activation_comparison with (security_invoker = true) as
select
    activation,
    n_layers,
    hidden_dim,
    lr,
    batch_size,
    epochs,
    seed,
    max_train_rows,
    include_protected,
    count(*)                       as runs,
    max(id)                        as latest_run_id,
    round(avg(train_accuracy)::numeric, 4) as train_accuracy,
    round(avg(val_accuracy)::numeric, 4)   as val_accuracy,
    round(avg(accuracy)::numeric, 4)       as test_accuracy,
    round(avg(f1)::numeric, 4)             as test_f1,
    round(avg(roc_auc)::numeric, 4)        as test_roc_auc,
    round(avg(brier)::numeric, 4)          as brier,
    round(avg(ece)::numeric, 4)            as ece,
    round(avg(final_loss)::numeric, 4)     as final_train_loss
from runs
where status = 'ok' and data_source = 'adult_income'
group by activation, n_layers, hidden_dim, lr, batch_size, epochs, seed,
         max_train_rows, include_protected;
