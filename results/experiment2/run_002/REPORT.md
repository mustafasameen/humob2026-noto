# Experiment 2: recovery by municipality

**PROVISIONAL reconstruction:** the final submission MD5 does not match the published reference. Actual: `e5032f9314ddfb61a0e76db9310932a8`; expected: `f19bfbd37dd859e0c0e928ffd0f83c5a`. The full-grid intermediate matches its published MD5, and the submission round-trip checks pass. The cause of the final-file mismatch is unresolved; halfway dates require a reference-file comparison. Observed anchor shares and ratios do not depend on the submission.
The preceding May–June reproduction scored 0.202910 to six decimals.

## Anchor comparison and reconstructed halfway dates

Shares are fractions of the complete daily flow total. Percentages below multiply those fractions by 100; the ratio is April mean share divided by January mean share.

| Municipality | January share (%) | April share (%) | April / January | 95% CI | First halfway date |
|---|---:|---:|---:|---|---|
| nanao | 1.2710 | 1.3455 | 1.0586 | [1.0380, 1.0819] | 20240213 |
| wajima | 0.2645 | 0.3324 | 1.2568 | [1.1965, 1.3220] | 20240213 |
| suzu | 0.2878 | 0.3038 | 1.0555 | [0.9998, 1.1257] | 20240201 |
| anamizu | 0.2640 | 0.3125 | 1.1836 | [1.1423, 1.2280] | 20240220 |
| noto_cho | 0.3001 | 0.3262 | 1.0868 | [1.0430, 1.1390] | 20240214 |
| shika | 0.4929 | 0.5327 | 1.0808 | [1.0447, 1.1213] | 20240213 |
| nakanoto | 0.7238 | 0.7021 | 0.9699 | [0.9523, 0.9881] | 20240201 |
| hakui | 0.1850 | 0.1930 | 1.0437 | [1.0105, 1.0766] | 20240219 |
| outside_grid | 13.5779 | 14.5858 | 1.0742 | [1.0424, 1.1064] | nan |

## Method and interpretation

- Cells are assigned with the existing `evidence.cell_to_muni` nearest-centroid mapping. These are evaluation-region cell groupings, not exact municipal boundaries or municipality-wide population estimates.
- A municipality numerator sums only its evaluation-region within-cell flows. The denominator sums every raw OD row, including within/outside-grid and cross-boundary flows.
- The outside-grid numerator is the within-bucket `-1_-1 -> -1_-1` flow. Cross-boundary flows enter the denominator but not this within-bucket numerator. `daily_totals.csv` also records all flows with an outside endpoint for audit.
- The raw total is constant to floating-point precision: 189190.250857142848. Each observed day uses its own all-row total; reconstruction uses their mean, never the sum of the model predictions.
- January 22–31 contributes 8 observed days; April contributes 28. Missing days are excluded from anchor means and resampling.
- The percentile interval resamples January and April days independently, with replacement, 4,000 times, seed 0. All municipalities share the same sampled date indices. Zero denominators produce explicit undefined statuses.
- Halfway means `(reconstructed share - January mean) / (April mean - January mean) >= 0.5`, including decreases. The first qualifying available day is reported; no smoothing or sustained-crossing condition is added.
- The two challenge-excluded dates, February 2 and March 5, remain blank. Dates therefore refer to the first of the 58 available reconstructed days. Missing observed days are also left blank.
- The submission has no outside-grid forecasts, so its reconstructed share and halfway date are unavailable. No residual or interpolated outside-bucket series is invented.
- Halfway dates describe the model path; they are not observed recovery dates. Ratio intervals cover sampling variation of observed anchor days, not reconstruction uncertainty.

## Outputs

- `municipality_summary.csv`: anchor shares, ratios and intervals, halfway levels/dates/statuses.
- `daily_shares.csv`: full-calendar observations, reconstructed shares and explicit missing statuses.
- `cell_to_municipality.csv`, `daily_totals.csv`: mapping and denominator audit.
- `municipality_shares.png` / `.pdf`: November–April daily-share figure.
- `commands.jsonl`, numbered logs, `packages.txt`, `run.json`, `checksums.json`, `verification.json`: provenance and validation.
- `submission.tsv` and `.fullgrid.tsv`, caches and forecasts are retained locally and excluded from Git by the repository rules.
