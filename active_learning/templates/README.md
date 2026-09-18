# Smoke-test templates

`eda_fast.in` and `force_fast.in` are the cheap level of theory
(wB97X-D/6-31G*, no non-local correlation grid) that
`scripts/smoke_test.sh --fast` runs, so a whole iteration of the loop finishes on an
interactive node in a couple of minutes instead of a quarter of an hour.

They exist to exercise the **plumbing** -- geometry to input to worker to output to merged
training frame -- and nothing they produce is training data: the numbers are not the
production level of theory, and an EDA decomposition at a different functional is not
comparable with one at wB97M-V/def2-TZVPD.

The `$molecule` block in each is a placeholder; the generator replaces it.
