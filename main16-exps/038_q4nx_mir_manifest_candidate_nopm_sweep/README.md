# Q4NX MIR Manifest Candidate NOPM Sweep

This experiment takes the weak source candidates from experiment 033 and runs a
strict direct-QKV numeric gate for each candidate. If experiment 033 has already
promoted those candidates to known-failed after this sweep, the script replays
the experiment-038 known-failed sites so the result remains reproducible.

Each candidate is a one-bundle `vmov` that the manifest currently classifies as
dead before use. The experiment replaces that bundle with the same MV-slot
`nopm` used by experiment 034:

```text
f8 4a 03 18
```

Run a smoke gate:

```bash
python main16-exps/038_q4nx_mir_manifest_candidate_nopm_sweep/run.py --max-sites 1 --max-cases 1
```

Run every manifest candidate against every synthetic direct-QKV case:

```bash
python main16-exps/038_q4nx_mir_manifest_candidate_nopm_sweep/run.py
```

Passing this experiment would be the first non-byte-copy source-bundle mutation
that survives the real NPU numeric gate.
