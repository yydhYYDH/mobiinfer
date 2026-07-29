# Requirements

The benchmark measures continuous mobile GUI-agent inference across consecutive
steps of the same task. It must compare full-prompt execution with history KV
reuse while keeping prompts, trajectory prefixes, model configuration, token
limit, and device conditions controlled.

Required checks are: process return code, timeout status, output-block count,
parseable action JSON, gold action match, token-boundary fallback count, logical
prompt tokens, actual prefill tokens, phase timing, and end-to-end timing.

Phone comparisons use OnePlus 13, fixed trajectory IDs, deterministic run order,
cooldown between cells, and at least three repeats for paper results. The pilot
is a controlled latency microbenchmark and must not be presented as a
dataset-wide accuracy result.
