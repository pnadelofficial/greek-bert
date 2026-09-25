# Results table

Auto-maintained by `scripts/runs.py::record_result`.
One row per (run, stage, model, metric, split, seed). Re-running a stage updates its row.

| run | stage | model | metric | value | split | seed | config_hash | git_sha | updated |
|---|---|---|---|---|---|---|---|---|---|
| _adhoc | convert | export | exported | ok |  | 22091997 | 0af0c31e27e8 | 508bb5f-dirty | 2026-09-23T19:07:12Z |
| arm1-aristo-continuation | pretrain | arm1-aristo-continuation | best_val_loss | 1.58181 | corpus-test | 22091997 | 2ed8975b5552 | fd5bd1b-dirty | 2026-09-25T00:11:38Z |
| arm1-aristo-continuation | pretrain | arm1-aristo-continuation | best_val_ppl | 4.86374 | corpus-test | 22091997 | 2ed8975b5552 | fd5bd1b-dirty | 2026-09-25T00:11:38Z |
| arm1-aristo-continuation | pretrain | arm1-aristo-continuation | final_val_loss | 1.58181 | corpus-test | 22091997 | 2ed8975b5552 | fd5bd1b-dirty | 2026-09-25T00:11:38Z |
| arm1-aristo-continuation | wsd | external-aristoBERTo | harmonia_best_val_acc | 0.809259 | glaux-val | 5-seed |  | fd5bd1b-dirty | 2026-09-25T01:10:17Z |
| arm1-aristo-continuation | wsd | external-aristoBERTo | harmonia_best_val_acc_std | 0.0105572 | glaux-val | 5-seed |  | fd5bd1b-dirty | 2026-09-25T01:10:18Z |
| arm1-aristo-continuation | wsd | external-aristoBERTo | kosmos_best_val_acc | 0.874909 | glaux-val | 5-seed |  | fd5bd1b-dirty | 2026-09-25T01:10:18Z |
| arm1-aristo-continuation | wsd | external-aristoBERTo | kosmos_best_val_acc_std | 0.0104763 | glaux-val | 5-seed |  | fd5bd1b-dirty | 2026-09-25T01:10:18Z |
