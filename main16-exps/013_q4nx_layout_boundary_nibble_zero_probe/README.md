# Q4NX Layout Boundary, Nibble, and Zero Probe

This experiment reuses the direct MyLM `0x1870` Q/K/V phase-body harness and
keeps only activation chunk 0 plus weight chunk 0 active.

It probes three remaining production-layout questions:

- the exact q4 data low/high output-half boundary;
- which payload lanes each nibble inside one q4 dword affects;
- whether zero/offset slots affect the observed payload and how they map to
  lanes.

The goal is not performance. The goal is an isolated NPU-observed numeric
contract before writing a MyLM-style source-assembly/codegen Q4NX body.
