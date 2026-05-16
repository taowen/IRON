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
| Runtime patch-site risk | 6 |
| Placement, ObjectFIFO, DMA, or L1 resource failure | 9, 11, 13, 14, 15, 16 |
| Full-ELF scratch layout changes correctness | 10 |
| Operator-specific numeric mismatch | 12 |
| Persistent phase ordering or timeout | 17 |
| Structured attention-score mismatch | 18, 20 |
| Persistent artifact should fail before runtime | 19 |
| PV/context compile or numeric failure | 14, 21, 22 |

### Runtime

- [1. Classify The Failure Boundary First](methods-runtime.md#1-classify-the-failure-boundary-first)
- [2. Probe pyxrt Capabilities](methods-runtime.md#2-probe-pyxrt-capabilities)
- [3. Inspect Artifacts Before Rerunning](methods-runtime.md#3-inspect-artifacts-before-rerunning)
- [5. Add Stage-Local Debug Drains](methods-runtime.md#5-add-stage-local-debug-drains)
- [6. Assert Runtime Patch Sites](methods-runtime.md#6-assert-runtime-patch-sites)
- [7. Repeat The Same Input](methods-runtime.md#7-repeat-the-same-input)
- [17. Treat Runtime Phase Assumptions As Suspect](methods-runtime.md#17-treat-runtime-phase-assumptions-as-suspect)

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

### Numeric

- [8. Use Local Reference And Full Reference Separately](methods-numeric.md#8-use-local-reference-and-full-reference-separately)
- [10. Treat Layout-Only Changes As Correctness Changes](methods-numeric.md#10-treat-layout-only-changes-as-correctness-changes)
- [12. Use Operator-Specific Tolerance](methods-numeric.md#12-use-operator-specific-tolerance)
- [18. Use Error Cardinality To Find Layout Bugs](methods-numeric.md#18-use-error-cardinality-to-find-layout-bugs)
- [21. Prove PV Inputs Before Changing The Context Kernel](methods-numeric.md#21-prove-pv-inputs-before-changing-the-context-kernel)
- [22. Match The Accumulation Boundary](methods-numeric.md#22-match-the-accumulation-boundary)
- [26. Rebuild Local References From The Actual FIFO Boundary](methods-numeric.md#26-rebuild-local-references-from-the-actual-fifo-boundary)
- [27. Test Approximation Kernels On The Model's Real Input Distribution](methods-numeric.md#27-test-approximation-kernels-on-the-models-real-input-distribution)

### Layout

- [11. Check ObjectFIFO Consumers In Generated MLIR](methods-layout.md#11-check-objectfifo-consumers-in-generated-mlir)
