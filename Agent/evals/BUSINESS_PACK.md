# Business test pack (Q01-Q52)

Run: `python -m Agent.evals.business_pack` (options `--engine-a 24 --engine-b 3 --cutoff-c 100 --cutoff-d 140 --only Q01,Q02`).
Writes `evals/results/business_pack_results.json` and `business_pack_report.md` (answers, trace IDs, citations, verdicts).

Conditional features implemented (Supported): fleet screening (Q06, Q07, Q17), trends and recent slope (Q05, Q09, Q10),
data quality (Q11), training-range and operating-setting checks (Q13, Q14), unit scenario (Q19), evidence block (Q30).
Not implemented by design, with a correct boundary answer: engine comparison (Q03), two-cutoff comparison (Q04), local
explanation (Q16), MTBF and availability (Q23), prediction interval (Q27).
Request-specific wording for refusals lives in `orchestration/boundaries.py`; add a document chunk and a boundary entry
rather than loosening a rule.
