-- 002_depth_status_threshold.sql
-- Extends the template schema from 001_init.sql. Apply it once in the Supabase
-- dashboard: SQL Editor -> New query -> paste -> Run. It is idempotent.
--
-- Apply this BEFORE deploying the API version that writes these columns.

-- ---------------------------------------------------------------------------
-- runs: network depth, run status, and the training-loss curve.
-- ---------------------------------------------------------------------------

-- Number of hidden layers. Rows written before this migration were trained with
-- the template's single-hidden-layer MLP, so they default to 1.
alter table runs add column if not exists n_layers integer not null default 1;

-- 'ok' or 'diverged'. A diverged run (NaN/Inf loss or a loss that ended higher
-- than it started) keeps NULL metrics instead of a sentinel number, so it can
-- never sort above, or be averaged in with, a real result.
alter table runs add column if not exists status text not null default 'ok';
alter table runs drop constraint if exists runs_status_check;
alter table runs add constraint runs_status_check
    check (status in ('ok', 'diverged'));

-- Metrics must be nullable for diverged runs (and roc_auc is also NULL when the
-- test split contains only one class, where AUC is undefined).
alter table runs alter column accuracy  drop not null;
alter table runs alter column precision drop not null;
alter table runs alter column recall    drop not null;
alter table runs alter column f1        drop not null;
alter table runs alter column roc_auc   drop not null;

-- Metrics are present exactly when the run is ok.
alter table runs drop constraint if exists runs_metrics_match_status;
alter table runs add constraint runs_metrics_match_status check (
    (status = 'ok'       and accuracy is not null and precision is not null
                         and recall   is not null and f1        is not null)
 or (status = 'diverged' and accuracy is null     and precision is null
                         and recall   is null     and f1        is null
                         and roc_auc  is null)
);

-- Per-epoch training BCE loss (for convergence plots) and its last value.
alter table runs add column if not exists final_loss   double precision;
alter table runs add column if not exists loss_history jsonb not null default '[]';

create index if not exists idx_runs_status on runs (status);

-- ---------------------------------------------------------------------------
-- predictions: the decision threshold used for each logged prediction, so the
-- audit log records how every label was produced.
-- ---------------------------------------------------------------------------
alter table predictions add column if not exists threshold double precision
    not null default 0.5;
alter table predictions drop constraint if exists predictions_threshold_range;
alter table predictions add constraint predictions_threshold_range
    check (threshold > 0 and threshold < 1);
