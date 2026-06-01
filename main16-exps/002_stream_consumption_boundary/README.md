# 002 Stream Consumption Boundary

This experiment turns the MyLM main16 dispatcher header sequence into a stream
consumption contract.

For each phase boundary, it compares:

- exact stream chunks: should produce the requested records;
- optionally, one chunk underfed: should time out waiting for the final requested
  record.

Expected chunk counts:

```text
Q/K/V:    12 records,  192 chunks
O:        20 records,  320 chunks cumulative
up/gate:  68 records, 1088 chunks cumulative
down:     76 records, 1472 chunks cumulative
```

Run:

```bash
./run.py
```

Underfed runs are intentionally timeout-based and can leave the raw-core runtime
session unsuitable for immediate follow-up runs. Run them separately:

```bash
./run.py --include-underfed --max-scenarios 2
```

Recover the pinned driver after an underfed timeout probe:

```bash
sudo systemctl restart amdxdna-pinned.service
```

For a staged run:

```bash
./run.py --max-scenarios 2
```
