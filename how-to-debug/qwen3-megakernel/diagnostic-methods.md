<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Diagnostic Methods

Use these entries after choosing a symptom from `symptoms.md`. Detailed
methods are split by diagnostic boundary.

## Method Selection

| Failure class | Use methods |
| --- | --- |
| Environment or Python/XRT binding mismatch | 1, 2 |
| Cached build artifacts or stale graph | 3 |
| Runtime BO metadata crash | 4 |
| Wrong final token or stage value | 5, 7, 8 |
| Runtime patch-site risk | 6, 48 |
| Placement, ObjectFIFO, DMA, or L1 resource failure | 9, 11, 13, 14, 15, 16, 44, 45, 47 |
| Phase packet layout or external-kernel ABI drift | 11, 50, 51 |
| External AIE kernel C++ compile/API failure | 43 |
| Full-ELF scratch layout changes correctness | 10 |
| Operator-specific numeric mismatch | 12 |
| Persistent phase ordering or timeout | 17 |
| Structured attention-score mismatch | 18, 20 |
| Persistent artifact should fail before runtime | 19 |
| PV/context compile or numeric failure | 14, 21, 22 |
| Multi-layer tensor handoff or diagnostic serialization crash | 31 |
| Composed checkpoint fails but standalone producer may pass | 30 |
| Decode is correct but much slower than NPU time suggests | 32 |
| Decode is correct and NPU time dominates | 35, 41, 42 |
| Column-sharded graph is correct but slower | 35, 41, 42 |
| Decode is correct but compile still happens per position | 6, 46 |
| Packed weight artifact, offset, or XRT sub-buffer risk | 33, 34 |
| Sweep script fails after loading many xclbins | 49 |
| New CLI stage fails before MLIR or NPU work | 50 |

### Runtime

- [1. Classify The Failure Boundary First](methods-runtime.md#1-classify-the-failure-boundary-first)
- [2. Probe pyxrt Capabilities](methods-runtime.md#2-probe-pyxrt-capabilities)
- [3. Inspect Artifacts Before Rerunning](methods-runtime.md#3-inspect-artifacts-before-rerunning)
- [5. Add Stage-Local Debug Drains](methods-runtime.md#5-add-stage-local-debug-drains)
- [6. Assert Runtime Patch Sites](methods-runtime.md#6-assert-runtime-patch-sites)
- [7. Repeat The Same Input](methods-runtime.md#7-repeat-the-same-input)
- [17. Treat Runtime Phase Assumptions As Suspect](methods-runtime.md#17-treat-runtime-phase-assumptions-as-suspect)
- [31. Clone XRT Tensor Views Before Crossing Debug Boundaries](methods-runtime.md#31-clone-xrt-tensor-views-before-crossing-debug-boundaries)
- [32. Split Wall Time From NPU Time](methods-runtime.md#32-split-wall-time-from-npu-time)
- [33. Prove Packed Weight BO Slices With Token Match](methods-runtime.md#33-prove-packed-weight-bo-slices-with-token-match)
- [42. Run A Phase Sensitivity Probe Before Widening](methods-runtime.md#42-run-a-phase-sensitivity-probe-before-widening)
- [48. Treat Control Packets As A Last-Resort Runtime Patch Path](methods-runtime.md#48-treat-control-packets-as-a-last-resort-runtime-patch-path)
- [49. Cleanup Runtime Between Independent Artifact Sweeps](methods-runtime.md#49-cleanup-runtime-between-independent-artifact-sweeps)

### Static

- [4. Compare runtime_sequence With main_kernels.json](methods-static.md#4-compare-runtimesequence-with-mainkernelsjson)
- [9. Read aiecc Resource Errors As Graph Errors](methods-static.md#9-read-aiecc-resource-errors-as-graph-errors)
- [13. Count Tile FIFO Inputs Before Changing Kernels](methods-static.md#13-count-tile-fifo-inputs-before-changing-kernels)
- [14. Read L1 MemoryMap Literally](methods-static.md#14-read-l1-memorymap-literally)
- [15. Inspect DMA Task Count, Not Just TAP Correctness](methods-static.md#15-inspect-dma-task-count-not-just-tap-correctness)
- [16. Validate TAP Against NPU BD Limits](methods-static.md#16-validate-tap-against-npu-bd-limits)
- [19. Run Persistent Artifact Preflight](methods-static.md#19-run-persistent-artifact-preflight)
- [20. Inspect Repeated ObjectFIFO Acquire Lowering](methods-static.md#20-inspect-repeated-objectfifo-acquire-lowering)
- [23. Check Producer Endpoints Before Reading Placer Errors As Resource Errors](methods-static.md#23-check-producer-endpoints-before-reading-placer-errors-as-resource-errors)
- [24. Count Workers Against The Actual Placer Budget](methods-static.md#24-count-workers-against-the-actual-placer-budget)
- [25. Optional Debug Streams Need One Boolean](methods-static.md#25-optional-debug-streams-need-one-boolean)
- [34. Validate Packed Weight Artifact Before Runtime](methods-static.md#34-validate-packed-weight-artifact-before-runtime)
- [35. Probe Real Graph Column Scaling](methods-static.md#35-probe-real-graph-column-scaling)
- [41. Estimate Static Work Before Choosing A Widening Target](methods-static.md#41-estimate-static-work-before-choosing-a-widening-target)
- [43. Read AIE API Compile Errors As Kernel-Boundary Evidence](methods-static.md#43-read-aie-api-compile-errors-as-kernel-boundary-evidence)
- [44. Full AIECC After ObjectFIFO Shape Changes](methods-static.md#44-full-aiecc-after-objectfifo-shape-changes)
- [45. Use ObjectFifo Split To Reduce Runtime Endpoints](methods-static.md#45-use-objectfifo-split-to-reduce-runtime-endpoints)
- [46. Diff Position Artifacts Before Choosing Patch Or Buckets](methods-static.md#46-diff-position-artifacts-before-choosing-patch-or-buckets)
- [47. Check ObjectFIFO Object Alignment After Metadata Tails](methods-static.md#47-check-objectfifo-object-alignment-after-metadata-tails)
- [50. Run Compileall Before NPU Debugging](methods-static.md#50-run-compileall-before-npu-debugging)

### Numeric

- [8. Use Local Reference And Full Reference Separately](methods-numeric.md#8-use-local-reference-and-full-reference-separately)
- [10. Treat Layout-Only Changes As Correctness Changes](methods-numeric.md#10-treat-layout-only-changes-as-correctness-changes)
- [12. Use Operator-Specific Tolerance](methods-numeric.md#12-use-operator-specific-tolerance)
- [18. Use Error Cardinality To Find Layout Bugs](methods-numeric.md#18-use-error-cardinality-to-find-layout-bugs)
- [21. Prove PV Inputs Before Changing The Context Kernel](methods-numeric.md#21-prove-pv-inputs-before-changing-the-context-kernel)
- [22. Match The Accumulation Boundary](methods-numeric.md#22-match-the-accumulation-boundary)
- [26. Rebuild Local References From The Actual FIFO Boundary](methods-numeric.md#26-rebuild-local-references-from-the-actual-fifo-boundary)
- [27. Test Approximation Kernels On The Model's Real Input Distribution](methods-numeric.md#27-test-approximation-kernels-on-the-models-real-input-distribution)
- [28. Freeze The Full-Depth Prefill Reference For Prefix Ladders](methods-numeric.md#28-freeze-the-full-depth-prefill-reference-for-prefix-ladders)
- [29. Compare The Same Value Through Two Consumers](methods-numeric.md#29-compare-the-same-value-through-two-consumers)
- [30. Export A Boundary Bundle And Re-run A Smaller Operator](methods-numeric.md#30-export-a-boundary-bundle-and-re-run-a-smaller-operator)

### Layout

- [11. Check ObjectFIFO Consumers In Generated MLIR](methods-layout.md#11-check-objectfifo-consumers-in-generated-mlir)
- [51. Treat Packet Header Changes As ABI Changes](methods-layout.md#51-treat-packet-header-changes-as-abi-changes)
