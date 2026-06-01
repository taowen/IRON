# 001 Dispatcher Header Sequence

This experiment uses the exp132 dispatcher stub with `control=1` and observes
the MyLM main16 dispatcher record headers over longer phase prefixes.

Expected sequence:

```text
20 records,  320 stream chunks: 12 x 0x1 + 8 x 0x4
68 records, 1088 stream chunks: 12 x 0x1 + 8 x 0x4 + 48 x 0x8
76 records, 1472 stream chunks: 12 x 0x1 + 8 x 0x4 + 48 x 0x8 + 8 x 0x4
```

The full 76-record run needs more input chunks than `records * 16`, because
the down phase consumes 48 chunks per record.

Run:

```bash
./run.py
```

For a staged run:

```bash
./run.py --max-scenarios 1
```
