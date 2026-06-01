# Q4NX Stream Parity Probe

Experiment 013 resolved the q4 field layout for active chunk pair 0. This
experiment checks whether that nibble behavior is stable across the even
activation/weight chunk pairs that contribute to one record.

The key question is whether q4 nibble positions `0/2/4/6` are globally inactive
for the observed direct-QKV chunk path, or whether they feed a different stream
parity/chunk path.
