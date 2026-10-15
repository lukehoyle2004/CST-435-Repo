-- 001_init.sql -- See-Sense: CIFAR-10 CNN classifier + Grad-CAM.
-- Apply in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Safe to re-run (drops and recreates the See-Sense objects only).
--
-- Privacy invariant: NO raw image bytes are ever stored. An upload is reduced to
-- its SHA-256 hash, its dimensions, and the model's answer.

drop view  if exists model_comparison;
drop table if exists image_metadata;
drop table if exists predictions;
drop table if exists run_artifacts;   -- from the synthetic-shapes template
drop table if exists runs;
drop table if exists datasets;        -- from the synthetic-shapes template

-- ---------------------------------------------------------------------------
-- runs: one row per training run (hyperparameters, train/val/test metrics,
-- per-class report, curves, and a pointer to the weights file the API serves).
-- Trained offline, inserted by db/seed_runs.py.
-- ---------------------------------------------------------------------------
create table runs (
    run_id            text primary key,              -- 8-hex id from train.py
    name              text not null unique,          -- 'baseline' | 'improved'
    display_name      text not null,
    architecture      text not null,
    techniques        text[] not null default '{}',  -- what the improved run adds
    dataset           text not null default 'CIFAR-10',
    epochs            integer not null,
    batch_size        integer not null,
    lr                double precision not null,
    lr_schedule       text not null,
    weight_decay      double precision not null,
    label_smoothing   double precision not null,
    augment           jsonb not null,
    seed              integer not null,
    params            integer not null,
    train_images      integer not null,
    val_images        integer not null,
    test_images       integer not null,
    hardware          text,
    train_seconds     double precision,
    best_epoch        integer not null,
    train_loss        double precision not null,
    train_acc         double precision not null,     -- percent, at best_epoch
    val_loss          double precision not null,
    val_acc           double precision not null,
    test_acc          double precision not null,     -- single view, percent
    test_acc_tta      double precision not null,     -- flip TTA (what /predict serves)
    macro_f1          double precision not null,
    per_class         jsonb not null,                -- {class: {precision, recall, f1}}
    confusion_matrix  jsonb not null,                -- 10x10 counts, rows = true class
    history           jsonb not null,                -- per-epoch loss / accuracy / lr
    weights_path      text not null,
    created_at        timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- predictions: one row per /predict call.
-- ---------------------------------------------------------------------------
create table predictions (
    id                bigint generated always as identity primary key,
    image_sha256      text not null check (image_sha256 ~ '^[0-9a-f]{64}$'),
    predicted_label   integer not null check (predicted_label between 0 and 9),
    predicted_class   text not null,
    confidence        double precision not null check (confidence between 0 and 1),
    top5              jsonb not null,                -- [{class_name, probability}] x 5
    served_by_run_id  text not null references runs (run_id) on delete cascade,
    tta               boolean not null default true,
    latency_ms        double precision,
    created_at        timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- image_metadata: hash + dimensions + label. Never pixels.
-- ---------------------------------------------------------------------------
create table image_metadata (
    id                bigint generated always as identity primary key,
    prediction_id     bigint not null references predictions (id) on delete cascade,
    image_sha256      text not null,
    width             integer not null check (width > 0),
    height            integer not null check (height > 0),
    format            text,                          -- PNG / JPEG / ...
    size_bytes        integer,
    predicted_class   text not null,
    created_at        timestamptz not null default now()
);

create index idx_predictions_created_at on predictions (created_at desc);
create index idx_predictions_run        on predictions (served_by_run_id);
create index idx_predictions_sha        on predictions (image_sha256);
create index idx_image_metadata_pred    on image_metadata (prediction_id);

-- ---------------------------------------------------------------------------
-- model_comparison: the headline numbers, straight from SQL.
-- ---------------------------------------------------------------------------
create view model_comparison with (security_invoker = true) as
select r.name,
       r.display_name,
       r.run_id,
       r.techniques,
       r.epochs,
       r.batch_size,
       r.params,
       round(r.test_acc_tta::numeric, 2)                         as test_acc_tta,
       round(r.test_acc::numeric, 2)                             as test_acc,
       round(r.macro_f1::numeric, 4)                             as macro_f1,
       round(r.val_acc::numeric, 2)                              as val_acc,
       round((r.train_acc - r.val_acc)::numeric, 2)              as generalization_gap,
       round((r.test_acc_tta - b.test_acc_tta)::numeric, 2)      as gain_over_baseline,
       (select count(*) from predictions p where p.served_by_run_id = r.run_id) as served_predictions
from runs r
cross join (select test_acc_tta from runs where name = 'baseline') b;

-- ---------------------------------------------------------------------------
-- Row Level Security. The API writes with the SERVICE-ROLE key (bypasses RLS).
-- The Streamlit UI reads with the public ANON key: SELECT only, no writes.
-- predictions/image_metadata hold only hashes + labels, so they are safe to
-- show on the public Recent Classifications wall.
-- ---------------------------------------------------------------------------
alter table runs           enable row level security;
alter table predictions    enable row level security;
alter table image_metadata enable row level security;

create policy "anon read runs"           on runs           for select to anon using (true);
create policy "anon read predictions"    on predictions    for select to anon using (true);
create policy "anon read image_metadata" on image_metadata for select to anon using (true);

grant select on model_comparison to anon;
