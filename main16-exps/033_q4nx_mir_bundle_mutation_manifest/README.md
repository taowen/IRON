# Q4NX MIR Bundle Mutation Manifest

This experiment turns the byte-exact MyLM hot-loop replay into a bundle-level
mutation manifest.

It records, for each hot-loop source or padding bundle:

- address and encoded bytes;
- op and semantic classes;
- cell-level defs/uses;
- whether each destination cell is used, overwritten, or terminal-dead;
- mutation class.

Run:

```bash
python3 main16-exps/033_q4nx_mir_bundle_mutation_manifest/run.py
```

The output is static analysis only. It does not replace the NPU numeric gate;
it chooses candidates that later experiments can test with direct-QKV payload
comparison.

Current model notes:

- `vups.4x` uses the measured transfer facts from experiments 036 and 037:
  `dm` updates all quadrants, `cml` updates only the low half, and `cmh`
  updates only the high half.
- Experiments 034 and 038 failed every current `vmov -> nopm` source candidate
  on the zero/offset direct-QKV numeric gate, so those bundles are tracked as
  known failed source mutations instead of being reported as weak candidates.
- `lfh*` writes are treated as external live-out because the hot-loop range
  ends before the surrounding phase body record emit is fully modeled.
