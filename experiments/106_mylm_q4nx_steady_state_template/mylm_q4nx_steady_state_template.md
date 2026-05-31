# MyLM Q4NX Steady-State Template

Parser source: `/var/home/taowen/projects/IRON/experiments/104_mylm_q4nx_pipeline_schedule/run.py`

This report extracts the repeatable middle of the MyLM Q4NX software
pipeline. It is intended as input for an assembly generator, not as a
replacement implementation.

## Group Identity

| Group | Slots | Text Hash | Op Hash | Matches Group1 Text | Matches Group1 Ops |
| ---: | ---: | --- | --- | --- | --- |
| 0 | 192 | `3440f93f9c98c0ba` | `49784ab7d84bfb00` | False | False |
| 1 | 189 | `debf92a97adcdff2` | `6f6d384cf442b675` | True | True |
| 2 | 189 | `debf92a97adcdff2` | `6f6d384cf442b675` | True | True |
| 3 | 189 | `debf92a97adcdff2` | `6f6d384cf442b675` | True | True |
| 4 | 189 | `debf92a97adcdff2` | `6f6d384cf442b675` | True | True |
| 5 | 189 | `debf92a97adcdff2` | `6f6d384cf442b675` | True | True |
| 6 | 190 | `254ee556c924f876` | `29a3ec713c50a10f` | False | False |
| 7 | 205 | `de55c8d50675c439` | `fee360e98322cefc` | False | False |

## Generator Shape

```text
fill(group0)
steady_template(group1) * 5  # groups 1..5 are text-identical
pre_drain(group6)
drain(group7)
```

## Steady Vext Lane Order

```text
0x0, 0x1, 0x2, 0x3, 0x8, 0x4, 0x5, 0x6, 0x7, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x10, 0x15, 0x11, 0x12, 0x18, 0x13, 0x14, 0x16, 0x17, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
```

The lane order is not simply 0..31. Preserving this order matters
because it is interleaved with unpack/upshift/convert/MAC work that
fills vector-load and storeback latency.

## Non-Steady Diffs

### `group0-fill` vs `group1`

| Kind | Base Range | Candidate Range | Base Text | Candidate Text |
| --- | --- | --- | --- | --- |
| insert | `0:0` | `0:1` | - | 0: `vlda	 x8, [p0], #0x40` |
| replace | `1:2` | `2:20` | 1: `vmac.f	dm1, dm1, x9, x10, r4` | 2: `lshl	 r18, r19, r2`<br>3: `add.nc	r16, r17, r1`<br>4: `vlda	 x7, [p0], #0x40`<br>5: `vunpack	x9, wl8, unpacksign0`<br>6: `lshl	 r20, r19, r3`<br>7: `add.nc	p5, r17, r18`<br>8: `vlda	 x0, [p0], #0x40`<br>9: `vunpack	x8, wh8, unpacksign0`<br>10: `movx	r19, #0x10`<br>11: `mov	dj0, r20`<br>12: `vlda	 lfh0, [p2, dj0]`<br>13: `vunpack	x5, wl7, unpacksign0`<br>14: `add.nc	p4, r18, r16`<br>15: `lda.s16	 r7, [p3], #0x2`<br>16: `vunpack	x10, wh7, unpacksign0`<br>17: `vunpack	x1, wl0, unpacksign0`<br>18: `nop`<br>19: `vextbcst.16	 x9, x11, #0x1` |
| replace | `5:6` | `23:25` | 5: `vmac.f	dm3, dm1, x4, x0, r4` | 23: `vadd	dm2, dm2, dm0, r0`<br>24: `vextbcst.16	 x7, x11, #0x0` |
| delete | `7:10` | `26:26` | 7: `vadd	dm2, dm2, dm0, r0`<br>8: `vmac.f	dm3, dm3, x3, x11, r4`<br>9: `vextbcst.16	 x7, x11, #0x0` | - |
| delete | `11:15` | `27:27` | 11: `vextbcst.16	 x9, x11, #0x1`<br>12: `nop`<br>13: `nop`<br>14: `vunpack	x1, wl0, unpacksign0` | - |
| delete | `16:17` | `28:28` | 16: `vconv.bf16.fp32	 wl4, bmll3` | - |
| insert | `20:20` | `31:32` | - | 31: `vmov	bmhh2, bmhh1` |
| replace | `21:24` | `33:35` | 21: `vmac.f	dm4, dm4, x4, x2, r4`<br>22: `nop`<br>23: `vmov	bmhh2, bmhh1` | 33: `vadd	dm4, dm1, dm0, r0`<br>34: `vconv.bf16.fp32	 x3, cmh2` |
| replace | `25:28` | `36:38` | 25: `vmac.f	dm4, dm4, x6, x1, r4`<br>26: `vconv.bf16.fp32	 x3, cmh2`<br>27: `nop` | 36: `vmov	bmhl1, bmhl3`<br>37: `vsub.f	dm2, dm4, dm0, r5` |
| replace | `29:36` | `39:40` | 29: `vadd	dm4, dm1, dm0, r0`<br>30: `nop`<br>31: `vmov	lfh0, bmll4`<br>32: `vsub.f	dm2, dm4, dm0, r5`<br>33: `nop`<br>34: `vmov	bmhl1, bmhl3`<br>35: `nop` | 39: `vmov	bmhh1, bmhh3` |
| replace | `37:38` | `41:42` | 37: `vmov	bmhh1, bmhh3` | 41: `vconv.bf16.fp32	 x8, cmh1` |
| delete | `39:40` | `43:43` | 39: `vconv.bf16.fp32	 x8, cmh1` | - |
| replace | `139:140` | `142:144` | 139: `nop` | 142: `vmov	bmll4, lfh0`<br>143: `vmov	bmhh1, bmhh4` |
| insert | `141:141` | `145:146` | - | 145: `vmac.f	dm3, dm3, x3, x8, r4` |
| replace | `142:144` | `147:148` | 142: `vmac.f	dm3, dm3, x3, x8, r4`<br>143: `vmov	bmhh1, bmhh4` | 147: `vmov	wl2, wh2` |
| delete | `145:146` | `149:149` | 145: `vmov	wl2, wh2` | - |
| replace | `163:164` | `166:167` | 163: `vmov	bmll4, lfh0` | 166: `nop` |

### `group6-pre-drain` vs `group1`

| Kind | Base Range | Candidate Range | Base Text | Candidate Text |
| --- | --- | --- | --- | --- |
| replace | `169:170` | `169:170` | 169: `nop` | 169: `mov	r21, p3` |
| insert | `178:178` | `178:179` | - | 178: `add.nc	p3, r21, #-0x10` |

### `group7-drain` vs `group1`

| Kind | Base Range | Candidate Range | Base Text | Candidate Text |
| --- | --- | --- | --- | --- |
| insert | `2:2` | `2:3` | - | 2: `paddb	 [p1], #-0x200` |
| replace | `139:140` | `140:142` | 139: `nop` | 140: `vmov	bmll4, lfh0`<br>141: `vmov	bmhh1, bmhh4` |
| insert | `141:141` | `143:144` | - | 143: `vmac.f	dm3, dm3, x3, x8, r4` |
| replace | `142:144` | `145:146` | 142: `vmac.f	dm3, dm3, x3, x8, r4`<br>143: `vmov	bmhh1, bmhh4` | 145: `vmov	wl2, wh2` |
| delete | `145:146` | `147:147` | 145: `vmov	wl2, wh2` | - |
| insert | `148:148` | `149:152` | - | 149: `vmov	bmhl1, bmhl2`<br>150: `vconv.bf16.fp32	 x1, cmh1`<br>151: `vmov	bmhh1, bmhh2` |
| replace | `149:150` | `153:155` | 149: `vconv.bf16.fp32	 x1, cmh1` | 153: `vmac.f	dm3, dm3, x2, x4, r4`<br>154: `vconv.bf16.fp32	 x4, cmh1` |
| replace | `151:154` | `156:157` | 151: `vmov	bmhl1, bmhl2`<br>152: `vmac.f	dm3, dm3, x2, x4, r4`<br>153: `vmov	bmhh1, bmhh2` | 156: `vextbcst.16	 x2, x11, #0x1a` |
| delete | `155:157` | `158:158` | 155: `vextbcst.16	 x2, x11, #0x1a`<br>156: `vconv.bf16.fp32	 x4, cmh1` | - |
| replace | `163:165` | `164:165` | 163: `vmov	bmll4, lfh0`<br>164: `lda.s16	 r7, [p3], #0x2` | 164: `nop` |
| replace | `169:170` | `169:171` | 169: `nop` | 169: `mov	p5, r16`<br>170: `paddb	 [p5], #-0x200` |
| delete | `171:172` | `172:172` | 171: `vlda	 x8, [p0], #0x40` | - |
| replace | `177:178` | `177:178` | 177: `vldb	 x7, [p0], #0x40` | 177: `mov	r17, p5` |
| delete | `179:180` | `179:179` | 179: `vunpack	x5, wl7, unpacksign0` | - |
| replace | `181:182` | `180:181` | 181: `vunpack	x9, wl8, unpacksign0` | 180: `nop` |
| replace | `183:186` | `182:184` | 183: `vunpack	x8, wh8, unpacksign0`<br>184: `vunpack	x10, wh7, unpacksign0`<br>185: `vldb	 x0, [p0], #0x40` | 182: `nop`<br>183: `nop` |
| insert | `189:189` | `187:205` | - | 187: `vmac.f	dm1, dm1, x9, x10, r4`<br>188: `nop`<br>189: `nop`<br>190: `vmac.f	dm3, dm1, x4, x0, r4`<br>191: `nop`<br>192: `nop`<br>193: `vmac.f	dm3, dm3, x3, x11, r4`<br>194: `nop`<br>195: `vconv.bf16.fp32	 wl4, bmll3`<br>196: `nop`<br>197: `vmac.f	dm4, dm4, x4, x2, r4`<br>198: `nop`<br>199: `nop`<br>200: `vmac.f	dm4, dm4, x6, x1, r4`<br>201: `nop`<br>202: `vmov	lfh0, bmll4`<br>203: `nopa`<br>204: `nopb` |

## Steady Template Body

Group1 is the canonical steady-state template. The same instruction text
is repeated for groups 2, 3, 4, and 5.

| Index | Offset | Bundle Slot | Instruction |
| ---: | ---: | ---: | --- |
| 0 | `+0x0` | 0 | `vldb	 x11, [p1], #0x40` |
| 1 | `+0x0` | 1 | `vmac.f	dm1, dm1, x9, x10, r4` |
| 2 | `+0x8` | 0 | `vups.4x	dm2, x9, s0, upssign0` |
| 3 | `+0xc` | 0 | `vadd	dm1, dm2, dm0, r0` |
| 4 | `+0x10` | 0 | `vups.4x	dm2, x8, s0, upssign0` |
| 5 | `+0x10` | 1 | `vmac.f	dm3, dm1, x4, x0, r4` |
| 6 | `+0x18` | 0 | `vsub.f	dm1, dm1, dm0, r5` |
| 7 | `+0x1c` | 0 | `vadd	dm2, dm2, dm0, r0` |
| 8 | `+0x20` | 0 | `vmac.f	dm3, dm3, x3, x11, r4` |
| 9 | `+0x24` | 0 | `vextbcst.16	 x7, x11, #0x0` |
| 10 | `+0x24` | 1 | `vsub.f	dm3, dm2, dm0, r5` |
| 11 | `+0x2c` | 0 | `vextbcst.16	 x9, x11, #0x1` |
| 12 | `+0x30` | 0 | `nop` |
| 13 | `+0x32` | 0 | `nop` |
| 14 | `+0x34` | 0 | `vunpack	x1, wl0, unpacksign0` |
| 15 | `+0x34` | 1 | `vmov	bmll2, bmll1` |
| 16 | `+0x3a` | 0 | `vconv.bf16.fp32	 wl4, bmll3` |
| 17 | `+0x3a` | 1 | `vmov	bmlh2, bmlh1` |
| 18 | `+0x42` | 0 | `vmov	bmhl2, bmhl1` |
| 19 | `+0x46` | 0 | `vconv.bf16.fp32	 x5, cml2` |
| 20 | `+0x46` | 1 | `vups.4x	dm1, x5, s0, upssign0` |
| 21 | `+0x46` | 2 | `vmac.f	dm4, dm4, x4, x2, r4` |
| 22 | `+0x50` | 0 | `nop` |
| 23 | `+0x52` | 0 | `vmov	bmhh2, bmhh1` |
| 24 | `+0x56` | 0 | `vmov	wl6, wh5` |
| 25 | `+0x56` | 1 | `vmac.f	dm4, dm4, x6, x1, r4` |
| 26 | `+0x5e` | 0 | `vconv.bf16.fp32	 x3, cmh2` |
| 27 | `+0x62` | 0 | `nop` |
| 28 | `+0x64` | 0 | `vmov	wl2, wh3` |
| 29 | `+0x68` | 0 | `vadd	dm4, dm1, dm0, r0` |
| 30 | `+0x6c` | 0 | `nop` |
| 31 | `+0x6e` | 0 | `vmov	lfh0, bmll4` |
| 32 | `+0x6e` | 1 | `vsub.f	dm2, dm4, dm0, r5` |
| 33 | `+0x76` | 0 | `nop` |
| 34 | `+0x78` | 0 | `vmov	bmhl1, bmhl3` |
| 35 | `+0x7c` | 0 | `nop` |
| 36 | `+0x7e` | 0 | `vldb	 x6, [p0], #0x40` |
| 37 | `+0x7e` | 1 | `vmov	bmhh1, bmhh3` |
| 38 | `+0x84` | 0 | `vmov	bmll1, bmll3` |
| 39 | `+0x88` | 0 | `vconv.bf16.fp32	 x8, cmh1` |
| 40 | `+0x88` | 1 | `vups.4x	dm3, x10, s0, upssign0` |
| 41 | `+0x88` | 2 | `vadd	dm3, dm3, dm0, r0` |
| 42 | `+0x92` | 0 | `vextbcst.16	 x10, x11, #0x2` |
| 43 | `+0x92` | 1 | `vmul.f	dm3, x5, x7, r4` |
| 44 | `+0x9a` | 0 | `vmov	bmlh1, bmlh3` |
| 45 | `+0x9a` | 1 | `vsub.f	dm4, dm3, dm0, r5` |
| 46 | `+0xa2` | 0 | `vextbcst.16	 x7, x11, #0x3` |
| 47 | `+0xa6` | 0 | `vconv.bf16.fp32	 x4, cml1` |
| 48 | `+0xa6` | 1 | `vmov	bmhh1, bmhh2` |
| 49 | `+0xa6` | 2 | `vmac.f	dm3, dm3, x6, x9, r4` |
| 50 | `+0xb0` | 0 | `vextbcst.16	 x9, x11, #0x8` |
| 51 | `+0xb4` | 0 | `vmov	wl10, wh4` |
| 52 | `+0xb8` | 0 | `vmov	bmll1, bmll2` |
| 53 | `+0xb8` | 1 | `vmac.f	dm3, dm3, x3, x10, r4` |
| 54 | `+0xc0` | 0 | `vmov	bmll2, bmll4` |
| 55 | `+0xc4` | 0 | `vunpack	x1, wh0, unpacksign0` |
| 56 | `+0xc4` | 1 | `vextbcst.16	 x7, x11, #0x4` |
| 57 | `+0xca` | 0 | `vmov	bmlh1, bmlh2` |
| 58 | `+0xca` | 1 | `vmac.f	dm3, dm3, x2, x7, r4` |
| 59 | `+0xd2` | 0 | `vmov	bmlh2, bmlh4` |
| 60 | `+0xd6` | 0 | `vconv.bf16.fp32	 x5, cml1` |
| 61 | `+0xd6` | 1 | `vextbcst.16	 x4, x11, #0x5` |
| 62 | `+0xde` | 0 | `vmov	bmhl1, bmhl2` |
| 63 | `+0xde` | 1 | `vmac.f	dm3, dm3, x4, x7, r4` |
| 64 | `+0xe6` | 0 | `vmov	bmhl2, bmhl4` |
| 65 | `+0xea` | 0 | `vconv.bf16.fp32	 x2, cmh1` |
| 66 | `+0xea` | 1 | `vups.4x	dm4, x1, s0, upssign0` |
| 67 | `+0xea` | 2 | `vadd	dm1, dm4, dm0, r0` |
| 68 | `+0xf4` | 0 | `vextbcst.16	 x3, x11, #0x6` |
| 69 | `+0xf8` | 0 | `vmov	bmhh2, bmhh4` |
| 70 | `+0xf8` | 1 | `vsub.f	dm4, dm1, dm0, r5` |
| 71 | `+0x100` | 0 | `vextbcst.16	 x0, x11, #0x7` |
| 72 | `+0x100` | 1 | `vmac.f	dm4, dm3, x10, x4, r4` |
| 73 | `+0x108` | 0 | `vextbcst.16	 x4, x11, #0x9` |
| 74 | `+0x10c` | 0 | `vconv.bf16.fp32	 x8, cml2` |
| 75 | `+0x10c` | 1 | `vmov	wl3, wh8` |
| 76 | `+0x10c` | 2 | `vmov.d	dm1, dm4` |
| 77 | `+0x116` | 0 | `vextbcst.16	 x7, x11, #0xa` |
| 78 | `+0x116` | 1 | `vmac.f	dm4, dm4, x8, x3, r4` |
| 79 | `+0x11e` | 0 | `vextbcst.16	 x10, x11, #0xb` |
| 80 | `+0x11e` | 1 | `vmov.d	dm3, dm1` |
| 81 | `+0x126` | 0 | `vmov	wl3, wh2` |
| 82 | `+0x12a` | 0 | `vmac.f	dm4, dm4, x3, x0, r4` |
| 83 | `+0x12e` | 0 | `nop` |
| 84 | `+0x130` | 0 | `vmov	wl5, wh5` |
| 85 | `+0x134` | 0 | `vmov	wl9, wh8` |
| 86 | `+0x134` | 1 | `vmac.f	dm4, dm4, x5, x9, r4` |
| 87 | `+0x13c` | 0 | `vmov	bmhl1, bmhl3` |
| 88 | `+0x140` | 0 | `vmov	bmhh1, bmhh3` |
| 89 | `+0x144` | 0 | `vconv.bf16.fp32	 x1, cmh2` |
| 90 | `+0x144` | 1 | `vmov	bmll1, bmll3` |
| 91 | `+0x144` | 2 | `vmac.f	dm4, dm4, x5, x4, r4` |
| 92 | `+0x14e` | 0 | `vconv.bf16.fp32	 x0, cmh1` |
| 93 | `+0x14e` | 1 | `vups.4x	dm3, x1, s0, upssign0` |
| 94 | `+0x14e` | 2 | `vadd	dm2, dm3, dm0, r0` |
| 95 | `+0x158` | 0 | `vextbcst.16	 x4, x11, #0xc` |
| 96 | `+0x15c` | 0 | `vunpack	x4, wl6, unpacksign0` |
| 97 | `+0x15c` | 1 | `vmov	bmlh1, bmlh3` |
| 98 | `+0x15c` | 2 | `vmac.f	dm3, dm4, x2, x7, r4` |
| 99 | `+0x166` | 0 | `vunpack	x6, wh6, unpacksign0` |
| 100 | `+0x166` | 1 | `vextbcst.16	 x5, x11, #0xd` |
| 101 | `+0x166` | 2 | `vsub.f	dm2, dm2, dm0, r5` |
| 102 | `+0x170` | 0 | `vconv.bf16.fp32	 x2, cml1` |
| 103 | `+0x170` | 1 | `vextbcst.16	 x10, x11, #0xe` |
| 104 | `+0x178` | 0 | `vextbcst.16	 x3, x11, #0xf` |
| 105 | `+0x178` | 1 | `vmac.f	dm1, dm3, x3, x10, r4` |
| 106 | `+0x180` | 0 | `vextbcst.16	 x7, x11, #0x10` |
| 107 | `+0x184` | 0 | `vmov	wl8, wh1` |
| 108 | `+0x188` | 0 | `vextbcst.16	 x4, x11, #0x15` |
| 109 | `+0x188` | 1 | `vmac.f	dm3, dm1, x8, x4, r4` |
| 110 | `+0x190` | 0 | `vups.4x	dm4, x4, s0, upssign0` |
| 111 | `+0x190` | 1 | `vadd	dm4, dm4, dm0, r0` |
| 112 | `+0x198` | 0 | `vmov	wl5, wh2` |
| 113 | `+0x19c` | 0 | `vextbcst.16	 x9, x11, #0x11` |
| 114 | `+0x19c` | 1 | `vmac.f	dm3, dm3, x9, x5, r4` |
| 115 | `+0x1a4` | 0 | `vsub.f	dm4, dm4, dm0, r5` |
| 116 | `+0x1a8` | 0 | `vextbcst.16	 x1, x11, #0x12` |
| 117 | `+0x1ac` | 0 | `vmov	bmll1, bmll2` |
| 118 | `+0x1ac` | 1 | `vmac.f	dm3, dm3, x1, x10, r4` |
| 119 | `+0x1b4` | 0 | `vextbcst.16	 x10, x11, #0x18` |
| 120 | `+0x1b8` | 0 | `vmov	wl3, wh0` |
| 121 | `+0x1bc` | 0 | `vextbcst.16	 x8, x11, #0x13` |
| 122 | `+0x1bc` | 1 | `vmac.f	dm3, dm3, x8, x3, r4` |
| 123 | `+0x1c4` | 0 | `vmov	bmlh1, bmlh2` |
| 124 | `+0x1c8` | 0 | `vextbcst.16	 x7, x11, #0x14` |
| 125 | `+0x1cc` | 0 | `vconv.bf16.fp32	 x2, cml1` |
| 126 | `+0x1cc` | 1 | `vmov	bmhl1, bmhl2` |
| 127 | `+0x1cc` | 2 | `vmac.f	dm3, dm3, x2, x7, r4` |
| 128 | `+0x1d6` | 0 | `vmov	bmhh1, bmhh2` |
| 129 | `+0x1da` | 0 | `vups.4x	dm1, x6, s0, upssign0` |
| 130 | `+0x1da` | 1 | `vadd	dm2, dm1, dm0, r0` |
| 131 | `+0x1e2` | 0 | `vconv.bf16.fp32	 x5, cmh1` |
| 132 | `+0x1e2` | 1 | `vextbcst.16	 x6, x11, #0x16` |
| 133 | `+0x1e2` | 2 | `vmac.f	dm3, dm3, x5, x9, r4` |
| 134 | `+0x1ec` | 0 | `vextbcst.16	 x9, x11, #0x17` |
| 135 | `+0x1ec` | 1 | `vsub.f	dm2, dm2, dm0, r5` |
| 136 | `+0x1f4` | 0 | `vextbcst.16	 x0, x11, #0x19` |
| 137 | `+0x1f8` | 0 | `vmov	bmll1, bmll4` |
| 138 | `+0x1f8` | 1 | `vmac.f	dm3, dm3, x0, x1, r4` |
| 139 | `+0x200` | 0 | `nop` |
| 140 | `+0x202` | 0 | `vmov	wl8, wh5` |
| 141 | `+0x206` | 0 | `vmov	bmlh1, bmlh4` |
| 142 | `+0x206` | 1 | `vmac.f	dm3, dm3, x3, x8, r4` |
| 143 | `+0x20e` | 0 | `vmov	bmhh1, bmhh4` |
| 144 | `+0x212` | 0 | `vconv.bf16.fp32	 x3, cml1` |
| 145 | `+0x212` | 1 | `vmov	wl2, wh2` |
| 146 | `+0x21a` | 0 | `vmov	bmhl1, bmhl4` |
| 147 | `+0x21a` | 1 | `vmac.f	dm3, dm3, x2, x7, r4` |
| 148 | `+0x222` | 0 | `vmov	bmll1, bmll2` |
| 149 | `+0x226` | 0 | `vconv.bf16.fp32	 x1, cmh1` |
| 150 | `+0x226` | 1 | `vmov	bmlh1, bmlh2` |
| 151 | `+0x22e` | 0 | `vmov	bmhl1, bmhl2` |
| 152 | `+0x22e` | 1 | `vmac.f	dm3, dm3, x2, x4, r4` |
| 153 | `+0x236` | 0 | `vmov	bmhh1, bmhh2` |
| 154 | `+0x23a` | 0 | `vconv.bf16.fp32	 x5, cml1` |
| 155 | `+0x23a` | 1 | `vextbcst.16	 x2, x11, #0x1a` |
| 156 | `+0x242` | 0 | `vconv.bf16.fp32	 x4, cmh1` |
| 157 | `+0x242` | 1 | `vextbcst.16	 x6, x11, #0x1b` |
| 158 | `+0x242` | 2 | `vmac.f	dm2, dm3, x5, x6, r4` |
| 159 | `+0x24c` | 0 | `vextbcst.16	 x7, x11, #0x1c` |
| 160 | `+0x250` | 0 | `vmov	wl8, wh1` |
| 161 | `+0x254` | 0 | `vmov	wl9, wh5` |
| 162 | `+0x254` | 1 | `vmac.f	dm1, dm2, x8, x9, r4` |
| 163 | `+0x25c` | 0 | `vmov	bmll4, lfh0` |
| 164 | `+0x260` | 0 | `lda.s16	 r7, [p3], #0x2` |
| 165 | `+0x260` | 1 | `vmov	wl3, wh3` |
| 166 | `+0x266` | 0 | `vldb	 wl2, [p5], #0x40` |
| 167 | `+0x266` | 1 | `vextbcst.16	 x10, x11, #0x1d` |
| 168 | `+0x266` | 2 | `vmac.f	dm1, dm1, x3, x10, r4` |
| 169 | `+0x270` | 0 | `nop` |
| 170 | `+0x272` | 0 | `vextbcst.16	 x0, x11, #0x1e` |
| 171 | `+0x276` | 0 | `vlda	 x8, [p0], #0x40` |
| 172 | `+0x276` | 1 | `vldb	 wl6, [p4], #0x40` |
| 173 | `+0x276` | 2 | `vextbcst.16	 x11, x11, #0x1f` |
| 174 | `+0x276` | 3 | `vmac.f	dm1, dm1, x3, x0, r4` |
| 175 | `+0x282` | 0 | `vmov	wl3, wh4` |
| 176 | `+0x286` | 0 | `vbcst.16	 x1, r7` |
| 177 | `+0x28a` | 0 | `vldb	 x7, [p0], #0x40` |
| 178 | `+0x28a` | 1 | `vmac.f	dm1, dm1, x1, x2, r4` |
| 179 | `+0x292` | 0 | `vunpack	x5, wl7, unpacksign0` |
| 180 | `+0x296` | 0 | `nop` |
| 181 | `+0x298` | 0 | `vunpack	x9, wl8, unpacksign0` |
| 182 | `+0x298` | 1 | `vmac.f	dm1, dm1, x8, x6, r4` |
| 183 | `+0x2a0` | 0 | `vunpack	x8, wh8, unpacksign0` |
| 184 | `+0x2a4` | 0 | `vunpack	x10, wh7, unpacksign0` |
| 185 | `+0x2a8` | 0 | `vldb	 x0, [p0], #0x40` |
| 186 | `+0x2a8` | 1 | `vmac.f	dm1, dm1, x5, x7, r4` |
| 187 | `+0x2b0` | 0 | `nop` |
| 188 | `+0x2b2` | 0 | `nop` |

## Checks

- group1..5 text identity is the only repeatable steady-state template
  found in the shipped MyLM hot loop.
- group6 must be generated as its own pre-drain variant because it
  rewinds `p3` near the tail before entering group7.
- group0 and group7 are not variants to guess from opcode counts; they
  need explicit fill/drain templates.
