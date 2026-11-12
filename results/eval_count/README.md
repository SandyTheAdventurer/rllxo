# Batch child evaluations of GP-SR-RLC (2026-11-12)

`analysis/count_evals.py` re-ran a2c_t03 with seed 0 on the 63 held-out equations, on the Linux laptop with 4 workers,
counting the candidate children whose batch loss the semantic bias computes (`computed`; memoized repeats are
`cached` and not counted) and the parent pairs scored or over the 2,500-pair cap.

- Only 12 of 63 runs give the same final model as the held-out (PC) run, so these are statistically equivalent
  runs, not the held-out runs themselves.
- Median 3.9 million children per run (mean 5.0 M, range 0 to 14.0 M), about 23,000 per generation; 2.3% of
  pairs were over the cap. Nguyen median 2.6 M, Keijzer 2.8 M, SRSD 5.6 M.
- Against 60,000 full evaluations (300 per generation x 200) on 512 points, that is about 8x the point evaluations.
- Quoted in the paper, Section 8.4.
