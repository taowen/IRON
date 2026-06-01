# Q4NX Schedule DSL Resource Manifest

This experiment starts the layer above raw MIR.

The goal is not to invent a full compiler. It adds a tiny schedule DSL:

```text
Program -> Bundle -> Operation
```

The DSL still emits pre-bundled MIR and uses `llc` for final encoding. The new
piece is a conservative bundle-resource manifest derived from the byte-exact
MyLM hot loop:

- classify every MIR operation into an opcode kind;
- record the opcode-kind tuple for every original bundle;
- reject generated candidates whose bundle opcode tuple never appears in MyLM.

This is intentionally conservative. It is useful because experiments 046 and
047 showed that guessing local bundle combinations either breaks NPU numeric
behavior or crashes the encoder. The first version should prove:

- the original MyLM schedule round-trips byte-exact through the DSL;
- the safe high-half swap from experiment 045 is accepted and encodes;
- the `vups.4x` advance candidates from experiments 046/047 are rejected before
  spending time on NPU gates.
