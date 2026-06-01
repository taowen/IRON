# 005 Non-Zero Payload Probe

This experiment keeps the direct Q/K/V phase-body entry from
`main16-exps/003_phase_body_direct_entry`, but changes the host input buffers.

Goal:

- run the MyLM `0x1870` Q/K/V body with the same MLIR route, DMA rings, locks,
  and direct-entry register profile already proven by experiment 003;
- compare compact record payloads for zero and deterministic non-zero
  activation/Q4NX weight streams;
- establish whether the raw phase body is numerically input-sensitive before
  decoding the exact MyLM Q4NX payload formula.

This is intentionally not a new dataflow experiment. It is a payload
observability experiment over the existing stable harness.

Runtime recovery after timeout:

```bash
sudo systemctl restart amdxdna-pinned.service
```

