```
test tests::bench_vec_dot_q8_ggml                     ... bench:         703 ns/iter (+/- 16)
test tests::bench_vec_dot_q8_naive                    ... bench:      10,834 ns/iter (+/- 190)
test tests::bench_vec_dot_q8_neon                     ... bench:         955 ns/iter (+/- 21)
test tests::bench_vec_dot_q8_neon_unrolled            ... bench:         712 ns/iter (+/- 4)
test tests::bench_vec_dot_q8_neon_unrolled_single_sum ... bench:         953 ns/iter (+/- 17)
test tests::bench_vec_dot_q8_neon_unrolled_vfma       ... bench:         702 ns/iter (+/- 8)
test tests::bench_vec_dot_q8_stdsimd                  ... bench:       3,063 ns/iter (+/- 19)
```

```
0000000000000c28 <__ZN8bench_q824vec_dot_q8_neon_unrolled17h9f95656789587f05E>:
     c28: d3451408     	ubfx	x8, x0, #5, #1
     c2c: ab401908     	adds	x8, x8, x0, lsr #6
     c30: 54000560     	b.eq	0xcdc <__ZN8bench_q824vec_dot_q8_neon_unrolled17h9f95656789587f05E+0xb4>
     c34: 91008829     	add	x9, x1, #0x22
     c38: 9100886a     	add	x10, x3, #0x22
     c3c: 6f00e400     	movi.2d	v0, #0000000000000000
     c40: 6f00e401     	movi.2d	v1, #0000000000000000
     c44: ad7f0d22     	ldp	q2, q3, [x9, #-0x20]
     c48: 3cc02124     	ldur	q4, [x9, #0x2]
     c4c: 3cc12125     	ldur	q5, [x9, #0x12]
     c50: ad7f1d46     	ldp	q6, q7, [x10, #-0x20]
     c54: 3cc02150     	ldur	q16, [x10, #0x2]
     c58: 3cc12151     	ldur	q17, [x10, #0x12]
     c5c: 6f00e412     	movi.2d	v18, #0000000000000000
     c60: 4e869452     	sdot
     c64: 4e879472     	sdot
     c68: 4e21da42     	scvtf.4s	v2, v18
     c6c: 7c5de123     	ldur	h3, [x9, #-0x22]
     c70: 1ee24066     	fcvt	s6, h3
     c74: 7c5de143     	ldur	h3, [x10, #-0x22]
     c78: 1ee24067     	fcvt	s7, h3
     c7c: 1e2708c3     	fmul	s3, s6, s7
     c80: 4f839042     	fmul.4s	v2, v2, v3[0]
     c84: 4e22d400     	fadd.4s	v0, v0, v2
     c88: 6f00e402     	movi.2d	v2, #0000000000000000
     c8c: 4e909482     	sdot
     c90: 4e9194a2     	sdot
     c94: 4e21d842     	scvtf.4s	v2, v2
     c98: 7d400123     	ldr	h3, [x9]
     c9c: 1ee24064     	fcvt	s4, h3
     ca0: 7d400143     	ldr	h3, [x10]
     ca4: 1ee24065     	fcvt	s5, h3
     ca8: 1e250883     	fmul	s3, s4, s5
     cac: 4f839042     	fmul.4s	v2, v2, v3[0]
     cb0: 4e22d421     	fadd.4s	v1, v1, v2
     cb4: 91011129     	add	x9, x9, #0x44
     cb8: 9101114a     	add	x10, x10, #0x44
     cbc: f1000508     	subs	x8, x8, #0x1
     cc0: 54fffc21     	b.ne	0xc44 <__ZN8bench_q824vec_dot_q8_neon_unrolled17h9f95656789587f05E+0x1c>
     cc4: 6e20d400     	faddp.4s	v0, v0, v0
     cc8: 7e30d800     	faddp.2s	s0, v0
     ccc: 6e21d421     	faddp.4s	v1, v1, v1
     cd0: 7e30d821     	faddp.2s	s1, v1
     cd4: 1e212800     	fadd	s0, s0, s1
     cd8: d65f03c0     	ret
     cdc: 6f00e401     	movi.2d	v1, #0000000000000000
     ce0: 6f00e400     	movi.2d	v0, #0000000000000000
     ce4: 6e20d400     	faddp.4s	v0, v0, v0
     ce8: 7e30d800     	faddp.2s	s0, v0
     cec: 6e21d421     	faddp.4s	v1, v1, v1
     cf0: 7e30d821     	faddp.2s	s1, v1
     cf4: 1e212800     	fadd	s0, s0, s1
     cf8: d65f03c0     	ret
```

```
00000000000183bc <_ggml_vec_dot_q8_0_q8_0>:
   183bc: 6f00e401     	movi.2d	v1, #0000000000000000
   183c0: 6f00e400     	movi.2d	v0, #0000000000000000
   183c4: 7100801f     	cmp	w0, #0x20
   183c8: 540004cb     	b.lt	0x18460 <_ggml_vec_dot_q8_0_q8_0+0xa4>
   183cc: d2800008     	mov	x8, #0x0                ; =0
   183d0: 53057c09     	lsr	w9, w0, #5
   183d4: 910088aa     	add	x10, x5, #0x22
   183d8: 9100886b     	add	x11, x3, #0x22
   183dc: 6f00e400     	movi.2d	v0, #0000000000000000
   183e0: 6f00e401     	movi.2d	v1, #0000000000000000
   183e4: ad7f0d62     	ldp	q2, q3, [x11, #-0x20]
   183e8: 3cc02164     	ldur	q4, [x11, #0x2]
   183ec: 3cc12165     	ldur	q5, [x11, #0x12]
   183f0: ad7f1d46     	ldp	q6, q7, [x10, #-0x20]
   183f4: 3cc02150     	ldur	q16, [x10, #0x2]
   183f8: 3cc12151     	ldur	q17, [x10, #0x12]
   183fc: 6f00e412     	movi.2d	v18, #0000000000000000
   18400: 4e869452     	sdot
   18404: 4e879472     	sdot
   18408: 4e21da42     	scvtf.4s	v2, v18
   1840c: 7c5de163     	ldur	h3, [x11, #-0x22]
   18410: 1ee24063     	fcvt	s3, h3
   18414: 7c5de146     	ldur	h6, [x10, #-0x22]
   18418: 1ee240c6     	fcvt	s6, h6
   1841c: 1e260863     	fmul	s3, s3, s6
   18420: 4f831040     	fmla.4s	v0, v2, v3[0]
   18424: 6f00e402     	movi.2d	v2, #0000000000000000
   18428: 4e909482     	sdot
   1842c: 4e9194a2     	sdot
   18430: 4e21d842     	scvtf.4s	v2, v2
   18434: 7d400163     	ldr	h3, [x11]
   18438: 1ee24063     	fcvt	s3, h3
   1843c: 7d400144     	ldr	h4, [x10]
   18440: 1ee24084     	fcvt	s4, h4
   18444: 1e240863     	fmul	s3, s3, s4
   18448: 4f831041     	fmla.4s	v1, v2, v3[0]
   1844c: 91000908     	add	x8, x8, #0x2
   18450: 9101114a     	add	x10, x10, #0x44
   18454: 9101116b     	add	x11, x11, #0x44
   18458: eb09011f     	cmp	x8, x9
   1845c: 54fffc43     	b.lo	0x183e4 <_ggml_vec_dot_q8_0_q8_0+0x28>
   18460: 6e20d400     	faddp.4s	v0, v0, v0
   18464: 7e30d800     	faddp.2s	s0, v0
   18468: 6e21d421     	faddp.4s	v1, v1, v1
   1846c: 7e30d821     	faddp.2s	s1, v1
   18470: 1e212800     	fadd	s0, s0, s1
   18474: bd000020     	str	s0, [x1]
   18478: d65f03c0     	ret
```
