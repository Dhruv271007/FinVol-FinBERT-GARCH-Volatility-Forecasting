# Corrections and provenance

Audit date: 21 October 2026.

## Preserved material

All 41 root-level CSVs remain unchanged. The twelve original Python scripts are archived in `output/original_source_before_corrections.zip`. No credentials or raw news text are included in the public bundle.

## Modeling and evaluation

- Defined forecasts at the actual XNYS close plus five minutes, with daylight saving and early closes.
- Used complete preceding issuance windows and target-aligned sentiment; retained article counts.
- Deduplicated old cached news by date and headline; new downloads preserve article IDs and update metadata.
- Added observed news-envelope screening, excluded BRK-B's incomplete end coverage, and added a contiguous final-segment sensitivity after long news gaps.
- Computed log price differences without forward filling; stopped dropping usable returns solely for unavailable descriptive rolling volatility.
- Retained the signed additive GARCH-X equation, adding sufficient positivity restrictions and a 0.999 persistence cap.
- Replaced one-start box-only estimation with constrained deterministic multi-start fitting and feasibility checks.
- Carried the training-only initial variance into sequential test filtering.
- Used a common training-mean-centered squared-return proxy and added EWMA and constant benchmarks.
- Added consistent Bartlett HAC covariance, degenerate-input handling, Holm adjustment, and explicit exploratory inference labels.
- Replaced invalid or boundary-sensitive unconstrained custom Wald outputs with parameter and diagnostic reporting.
- Exported forecasts, parameters, losses, sample dates, convergence counts, boundary flags, and residual diagnostics.

## Software

- Centralized configuration and removed import-time downloads, fits, merges, and CSV overwrites from analysis entry points.
- Fixed the broken sentiment-module import; made FinBERT inference lazy with explicit label lookup and evaluation mode.
- Made the GPU check safe when no CUDA device exists and repaired the standalone plotting script.
- Added pinned analysis dependencies, optional dependencies, a run guide, input hashes, eight behavioral tests, and a reproducible report builder.
- Made integer timestamp comparisons explicit in nanoseconds to prevent pandas resolution mismatches.

## Repository cleanup (9 November 2026)

- Vectorised exchange-calendar close lookup and EWMA recursion; GARCH-X fitting now reuses the already-fitted GARCH baseline instead of refitting it. Outputs are bit-identical.
- `yfinancedownload.py` now actually offers an explicit `--download` path (to `output/refreshed_prices/`, never overwriting the cache).
- `verify_report.py` runs the unit tests itself rather than recording a hard-coded pass.
- `build_report.py` falls back to built-in Helvetica when Calibri is unavailable.
- Removed scratch/duplicate scripts (`temp.py`, `lag_sentiment.py`; both remain in the original-source archive), unrelated editor folders, caches, and regenerated temp images.

## Remaining limitations

The cache contains incomplete coverage, long gaps, unknown price-adjustment provenance, and no historical first-seen text archive. Existing sentiment scores were reused rather than re-inferred. Credentialed collection and optional inference paths were not executed. The report distinguishes these limitations from completed software repairs.
