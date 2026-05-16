<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIE Kernel Compile Failures

Used during SageAttention bring-up.

## Symptom

After changing dequantization from scalar to vector code, Peano/clang failed:

```text
error: no matching member function for call to 'from_vector'
```

The failing code called `acc.from_vector(scaled, 0)`.

## Cause

For this expression:

```cpp
auto scaled = aie::mul(qk_f32, scale);
```

`scaled` is not the plain vector type expected by `acc.from_vector(...)`.

## Fix Used

Store the multiply result directly after converting it to bf16:

```cpp
auto qk_i32 = aie::load_v<16>(scratch_i32 + i);
auto qk_f32 = aie::to_float<float>(qk_i32, 0);
auto scaled = aie::mul(qk_f32, scale);
aie::store_v(logits_out + i, scaled.to_vector<bfloat16>());
```

The useful lookup was the local AIE API header:

```bash
grep -R "to_float" -n .venv/lib/python3.12/site-packages/mlir_aie/include/aie_api | head
```
