# Main16 Experiments

This directory is the working notebook for learning and reproducing the MyLM
main16 raw core program.

Scope:

- understand the MyLM c2r2 main16 entry, dispatcher, phase bodies, Q4NX
  microkernel, record headers, locks, and tile-local control memory;
- keep public AMD/Xilinx references next to the reverse-engineering notes;
- keep new main16-only experiments here instead of scattering them through the
  top-level `experiments/` sequence;
- preserve existing runnable experiments `128..133` in place, because their
  scripts and generated reports already reference those paths.

Start here:

- [docs/public-references.md](docs/public-references.md)
- [docs/current-contract.md](docs/current-contract.md)
- [docs/experiment-index.md](docs/experiment-index.md)
- [docs/next-experiments.md](docs/next-experiments.md)

Runtime recovery:

- normal non-timeout probes should leave `xrt-smi examine` reporting
  `RyzenAI-npu4 / aie2p / 6x8`;
- timeout-based underfeed probes can dirty the pinned NPU driver state;
- recover the pinned driver with:

```bash
sudo systemctl restart amdxdna-pinned.service
```
