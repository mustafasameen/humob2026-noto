# Experiment 4 run index

- `run_001`: failed runner assertion on a reference table containing all arms; retained.
- `run_002`: failed before scoring because the raw TSV was moved after Task 2; retained. The sweep itself only needs the prepared cache, so the runner no longer hashes the raw TSV.
- `run_003`: evaluated all 16 windows and reproduced May–June; first sweep stopped because the reference file contained all arms.
- `run_004`: passed a stale path to the same reference file; retained.
- `run_005`: complete. It copies the successful `run_003/reference` results, filters `final` for the strict sweep reference gate, runs all four grids, and saves every variant CSV, pooled comparison, commands, versions, source/input hashes, and report. Every submitted row passed the 1e-12 reference assertion.

A direct May–June `evaluate --set fm_diagonal_weight=0.25` check matched the sweep result to 1.1e-16. The code and scoring logic were not changed. `tests.log` records 27 passed, 1 skipped. `FINDINGS.md` holds draft result text. `artifact-checksums.json` covers the complete results directory, including failed attempts and post-run notes.
