Early tuning source hashes match the accompanying btmf-before-fallback-fix.json. The subsequent fix only changes the large-matrix fallback (pairs * rank squared > 100,000,000), never entered by these runs (1,395 pairs, ranks at most 20). The numerical update path is unchanged.

Runs carrying this earlier source hash:
- results/experiment1/tuning/btmf_rank5/may_jun/run.json
- results/experiment1/tuning/btmf_rank10/may_jun/run.json
