# Check after author's LF submission update

Author commit `06a2eaa9c5a03ad354b03712a50415204019e6fc` explicitly writes LF line endings. This branch merged it at `f4e547e`.

Command (using the preserved prepared caches and released forecast files):

```bash
.venv/bin/python -m humob26 submit --cache cache --forecasts forecasts/timesfm3_mlx --out results/experiment2/checksum_after_lf/submission.tsv > results/experiment2/checksum_after_lf/command.log 2>&1
md5 results/experiment2/checksum_after_lf/submission.tsv results/experiment2/checksum_after_lf/submission.fullgrid.tsv > results/experiment2/checksum_after_lf/md5.txt
.venv/bin/pip freeze > results/experiment2/checksum_after_lf/packages.txt
```

The final submission remains `e5032f9314ddfb61a0e76db9310932a8` (expected `f19bfbd37dd859e0c0e928ffd0f83c5a`). The full-grid intermediate still matches `be03ccbf5aaebd81a275b3a52953504f`, and the submission's internal checks pass. The new final file is byte-identical to `run_002/submission.tsv`. Converting only its LF line endings to CRLF would produce `c327498a534501d2dcfe27e496f66991`, also different from the published MD5.

The input dataset symlink to `/Users/mrunal/Downloads/humob2026-dataset.tsv` was broken at the time, so `run_003` retains that failed attempt. This follow-up used the caches already prepared from the original raw file. The dataset was subsequently restored and the full strict pipeline rerun as `run_004`; the mismatch persisted. The cause remains unresolved, and the existing halfway dates remain provisional.
