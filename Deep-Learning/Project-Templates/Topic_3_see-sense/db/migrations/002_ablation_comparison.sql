-- Leave-one-out ablations: each `no_*` run is the improved recipe with ONE technique removed,
-- trained with the same seed, epochs, batch size and split. The accuracy drop versus the full
-- improved run is that technique's contribution. Run after 001_init.sql, then `python -m db.seed_runs`.
--
--   select * from ablation_comparison;

drop view if exists ablation_comparison;

create view ablation_comparison with (security_invoker = true) as
select r.name,
       r.display_name,
       r.run_id,
       substr(r.name, 4)                                           as technique_removed,
       r.epochs,
       r.seed,
       round(r.test_acc_tta::numeric, 2)                           as test_acc_tta,
       round(r.macro_f1::numeric, 4)                               as macro_f1,
       round((r.train_acc - r.val_acc)::numeric, 2)                as generalization_gap,
       round((i.test_acc_tta - r.test_acc_tta)::numeric, 2)        as contribution_pts,
       rank() over (order by i.test_acc_tta - r.test_acc_tta desc) as contribution_rank
from runs r
cross join (select test_acc_tta from runs where name = 'improved') i
where r.name like 'no\_%'
order by contribution_pts desc;

grant select on ablation_comparison to anon, authenticated;
