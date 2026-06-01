# MyLM Main16 Dispatcher Stub Probe

- Status: `completed`
- Records per variant: `1`
- Variants attempted: `3`

## Results

| variant | control | status | unique headers | topology after |
| --- | ---: | --- | --- | --- |
| `dispatcher_stub_control_0` | `0` | `record_observed` | `['0x4']` | `6x8` |
| `dispatcher_stub_control_1` | `1` | `record_observed` | `['0x1']` | `6x8` |
| `dispatcher_stub_control_8` | `8` | `record_observed` | `['0x4']` | `6x8` |

## Interpretation

The caller-side control slot successfully selected the Q/K/V dispatcher path. This gives us an observable way to enter MyLM's `0x1870` body without guessing static data words.
`control=8` still emitted `0x4`, so this control point is not a general phase-id selector. It behaves as a QKV-vs-alternate gate: value `1` enters the `0x1870` Q/K/V body, while non-`1` values take the alternate `0x4` path.

## Runtime Output Preview

### dispatcher_stub_control_0

```text
load_begin
load_ok
run_ok
elapsed=0.0006029605865478516
result_type=XRTKernelResult
npu_time=556938
record=0x4,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0
```

### dispatcher_stub_control_1

```text
load_begin
load_ok
run_ok
elapsed=0.0005826950073242188
result_type=XRTKernelResult
npu_time=535688
record=0x1,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0
```

### dispatcher_stub_control_8

```text
load_begin
load_ok
run_ok
elapsed=0.0005464553833007812
result_type=XRTKernelResult
npu_time=493299
record=0x4,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0
```
