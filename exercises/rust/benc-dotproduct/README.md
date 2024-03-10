```
dotproduct_bench-2a79baec15c9b09c.dotproduct_bench.81ee4170704a15e0-cgu.0.rcgu.o:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000000000000 <ltmp0>:
       0: a9bf7bfd     	stp	x29, x30, [sp, #-0x10]!
       4: 910003fd     	mov	x29, sp
       8: b4000121     	cbz	x1, 0x2c <ltmp0+0x2c>
       c: d1000428     	sub	x8, x1, #0x1
      10: eb03011f     	cmp	x8, x3
      14: 54000822     	b.hs	0x118 <ltmp0+0x118>
      18: f100403f     	cmp	x1, #0x10
      1c: 540000c2     	b.hs	0x34 <ltmp0+0x34>
      20: d2800008     	mov	x8, #0x0                ; =0
      24: 2f00e400     	movi	d0, #0000000000000000
      28: 14000030     	b	0xe8 <ltmp0+0xe8>
      2c: 2f00e400     	movi	d0, #0000000000000000
      30: 14000038     	b	0x110 <ltmp0+0x110>
      34: 927cec28     	and	x8, x1, #0xfffffffffffffff0
      38: 91008009     	add	x9, x0, #0x20
      3c: 9100804a     	add	x10, x2, #0x20
      40: 2f00e400     	movi	d0, #0000000000000000
      44: aa0803eb     	mov	x11, x8
      48: ad7f0921     	ldp	q1, q2, [x9, #-0x20]
      4c: acc21123     	ldp	q3, q4, [x9], #0x40
      50: ad7f1945     	ldp	q5, q6, [x10, #-0x20]
      54: acc24147     	ldp	q7, q16, [x10], #0x40
      58: 6e25dc21     	fmul.4s	v1, v1, v5
      5c: 5e1c0425     	mov	s5, v1[3]
      60: 5e140431     	mov	s17, v1[2]
      64: 5e0c0432     	mov	s18, v1[1]
      68: 6e26dc42     	fmul.4s	v2, v2, v6
      6c: 5e1c0446     	mov	s6, v2[3]
      70: 5e140453     	mov	s19, v2[2]
      74: 5e0c0454     	mov	s20, v2[1]
      78: 6e27dc63     	fmul.4s	v3, v3, v7
      7c: 5e1c0467     	mov	s7, v3[3]
      80: 5e140475     	mov	s21, v3[2]
      84: 5e0c0476     	mov	s22, v3[1]
      88: 6e30dc84     	fmul.4s	v4, v4, v16
      8c: 5e1c0490     	mov	s16, v4[3]
      90: 5e140497     	mov	s23, v4[2]
      94: 5e0c0498     	mov	s24, v4[1]
      98: 1e212800     	fadd	s0, s0, s1
      9c: 1e322800     	fadd	s0, s0, s18
      a0: 1e312800     	fadd	s0, s0, s17
      a4: 1e252800     	fadd	s0, s0, s5
      a8: 1e222800     	fadd	s0, s0, s2
      ac: 1e342800     	fadd	s0, s0, s20
      b0: 1e332800     	fadd	s0, s0, s19
      b4: 1e262800     	fadd	s0, s0, s6
      b8: 1e232800     	fadd	s0, s0, s3
      bc: 1e362800     	fadd	s0, s0, s22
      c0: 1e352800     	fadd	s0, s0, s21
      c4: 1e272800     	fadd	s0, s0, s7
      c8: 1e242800     	fadd	s0, s0, s4
      cc: 1e382800     	fadd	s0, s0, s24
      d0: 1e372800     	fadd	s0, s0, s23
      d4: 1e302800     	fadd	s0, s0, s16
      d8: f100416b     	subs	x11, x11, #0x10
      dc: 54fffb61     	b.ne	0x48 <ltmp0+0x48>
      e0: eb01011f     	cmp	x8, x1
      e4: 54000160     	b.eq	0x110 <ltmp0+0x110>
      e8: d37ef50a     	lsl	x10, x8, #2
      ec: 8b0a0049     	add	x9, x2, x10
      f0: 8b0a000a     	add	x10, x0, x10
      f4: cb080028     	sub	x8, x1, x8
      f8: bc404541     	ldr	s1, [x10], #0x4
      fc: bc404522     	ldr	s2, [x9], #0x4
     100: 1e220821     	fmul	s1, s1, s2
     104: 1e212800     	fadd	s0, s0, s1
     108: f1000508     	subs	x8, x8, #0x1
     10c: 54ffff61     	b.ne	0xf8 <ltmp0+0xf8>
     110: a8c17bfd     	ldp	x29, x30, [sp], #0x10
     114: d65f03c0     	ret
     118: 90000002     	adrp	x2, 0x0 <ltmp0>
     11c: 91000042     	add	x2, x2, #0x0
     120: aa0303e0     	mov	x0, x3
     124: aa0303e1     	mov	x1, x3
     128: 94000000     	bl	0x128 <ltmp0+0x128>

000000000000012c <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E>:
     12c: a9bf7bfd     	stp	x29, x30, [sp, #-0x10]!
     130: 910003fd     	mov	x29, sp
     134: 2f00e400     	movi	d0, #0000000000000000
     138: f100103f     	cmp	x1, #0x4
     13c: 54000843     	b.lo	0x244 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x118>
     140: aa0003e8     	mov	x8, x0
     144: d342fc29     	lsr	x9, x1, #2
     148: eb03003f     	cmp	x1, x3
     14c: 54000429     	b.ls	0x1d0 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0xa4>
     150: d2800000     	mov	x0, #0x0                ; =0
     154: 9100210a     	add	x10, x8, #0x8
     158: 9100204b     	add	x11, x2, #0x8
     15c: eb01001f     	cmp	x0, x1
     160: 54000762     	b.hs	0x24c <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x120>
     164: eb03001f     	cmp	x0, x3
     168: 54000ae2     	b.hs	0x2c4 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x198>
     16c: 91000408     	add	x8, x0, #0x1
     170: eb03011f     	cmp	x8, x3
     174: 54000722     	b.hs	0x258 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x12c>
     178: 91000808     	add	x8, x0, #0x2
     17c: eb03011f     	cmp	x8, x3
     180: 540007e2     	b.hs	0x27c <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x150>
     184: 91000c08     	add	x8, x0, #0x3
     188: eb03011f     	cmp	x8, x3
     18c: 54000922     	b.hs	0x2b0 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x184>
     190: fc5f8141     	ldur	d1, [x10, #-0x8]
     194: fc5f8162     	ldur	d2, [x11, #-0x8]
     198: 2e22dc21     	fmul.2s	v1, v1, v2
     19c: 1e212800     	fadd	s0, s0, s1
     1a0: 5e0c0421     	mov	s1, v1[1]
     1a4: 1e212800     	fadd	s0, s0, s1
     1a8: fc410541     	ldr	d1, [x10], #0x10
     1ac: fc410562     	ldr	d2, [x11], #0x10
     1b0: 2e22dc21     	fmul.2s	v1, v1, v2
     1b4: 1e212800     	fadd	s0, s0, s1
     1b8: 5e0c0421     	mov	s1, v1[1]
     1bc: 1e212800     	fadd	s0, s0, s1
     1c0: 91001000     	add	x0, x0, #0x4
     1c4: f1000529     	subs	x9, x9, #0x1
     1c8: 54fffca1     	b.ne	0x15c <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x30>
     1cc: 1400001e     	b	0x244 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x118>
     1d0: d2800000     	mov	x0, #0x0                ; =0
     1d4: 9100210a     	add	x10, x8, #0x8
     1d8: 9100204b     	add	x11, x2, #0x8
     1dc: eb01001f     	cmp	x0, x1
     1e0: 54000362     	b.hs	0x24c <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x120>
     1e4: 91000408     	add	x8, x0, #0x1
     1e8: eb01011f     	cmp	x8, x1
     1ec: 54000402     	b.hs	0x26c <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x140>
     1f0: 91000808     	add	x8, x0, #0x2
     1f4: eb01011f     	cmp	x8, x1
     1f8: 540004c2     	b.hs	0x290 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x164>
     1fc: 91000c08     	add	x8, x0, #0x3
     200: eb01011f     	cmp	x8, x1
     204: 540004e2     	b.hs	0x2a0 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x174>
     208: fc5f8141     	ldur	d1, [x10, #-0x8]
     20c: fc5f8162     	ldur	d2, [x11, #-0x8]
     210: 2e22dc21     	fmul.2s	v1, v1, v2
     214: 1e212800     	fadd	s0, s0, s1
     218: 5e0c0421     	mov	s1, v1[1]
     21c: 1e212800     	fadd	s0, s0, s1
     220: fc410541     	ldr	d1, [x10], #0x10
     224: fc410562     	ldr	d2, [x11], #0x10
     228: 2e22dc21     	fmul.2s	v1, v1, v2
     22c: 1e212800     	fadd	s0, s0, s1
     230: 5e0c0421     	mov	s1, v1[1]
     234: 1e212800     	fadd	s0, s0, s1
     238: 91001000     	add	x0, x0, #0x4
     23c: f1000529     	subs	x9, x9, #0x1
     240: 54fffce1     	b.ne	0x1dc <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0xb0>
     244: a8c17bfd     	ldp	x29, x30, [sp], #0x10
     248: d65f03c0     	ret
     24c: 90000002     	adrp	x2, 0x0 <ltmp0>
     250: 91000042     	add	x2, x2, #0x0
     254: 94000000     	bl	0x254 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x128>
     258: 90000002     	adrp	x2, 0x0 <ltmp0>
     25c: 91000042     	add	x2, x2, #0x0
     260: aa0803e0     	mov	x0, x8
     264: aa0303e1     	mov	x1, x3
     268: 94000000     	bl	0x268 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x13c>
     26c: 90000002     	adrp	x2, 0x0 <ltmp0>
     270: 91000042     	add	x2, x2, #0x0
     274: aa0803e0     	mov	x0, x8
     278: 94000000     	bl	0x278 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x14c>
     27c: 90000002     	adrp	x2, 0x0 <ltmp0>
     280: 91000042     	add	x2, x2, #0x0
     284: aa0803e0     	mov	x0, x8
     288: aa0303e1     	mov	x1, x3
     28c: 94000000     	bl	0x28c <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x160>
     290: 90000002     	adrp	x2, 0x0 <ltmp0>
     294: 91000042     	add	x2, x2, #0x0
     298: aa0803e0     	mov	x0, x8
     29c: 94000000     	bl	0x29c <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x170>
     2a0: 90000002     	adrp	x2, 0x0 <ltmp0>
     2a4: 91000042     	add	x2, x2, #0x0
     2a8: aa0803e0     	mov	x0, x8
     2ac: 94000000     	bl	0x2ac <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x180>
     2b0: 90000002     	adrp	x2, 0x0 <ltmp0>
     2b4: 91000042     	add	x2, x2, #0x0
     2b8: aa0803e0     	mov	x0, x8
     2bc: aa0303e1     	mov	x1, x3
     2c0: 94000000     	bl	0x2c0 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x194>
     2c4: 90000002     	adrp	x2, 0x0 <ltmp0>
     2c8: 91000042     	add	x2, x2, #0x0
     2cc: aa0303e1     	mov	x1, x3
     2d0: 94000000     	bl	0x2d0 <__ZN16dotproduct_bench7dotprod16dot_naive_unroll17h24b09a75436b39d2E+0x1a4>

00000000000002d4 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E>:
     2d4: a9bf7bfd     	stp	x29, x30, [sp, #-0x10]!
     2d8: 910003fd     	mov	x29, sp
     2dc: f100103f     	cmp	x1, #0x4
     2e0: 540000a2     	b.hs	0x2f4 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x20>
     2e4: 2f00e400     	movi	d0, #0000000000000000
     2e8: 2f00e402     	movi	d2, #0000000000000000
     2ec: 2f00e401     	movi	d1, #0000000000000000
     2f0: 1400004c     	b	0x420 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x14c>
     2f4: aa0003e8     	mov	x8, x0
     2f8: d342fc29     	lsr	x9, x1, #2
     2fc: eb03003f     	cmp	x1, x3
     300: 540004c9     	b.ls	0x398 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0xc4>
     304: d2800000     	mov	x0, #0x0                ; =0
     308: 9100110a     	add	x10, x8, #0x4
     30c: 9100104b     	add	x11, x2, #0x4
     310: 2f00e400     	movi	d0, #0000000000000000
     314: 2f00e401     	movi	d1, #0000000000000000
     318: 2f00e402     	movi	d2, #0000000000000000
     31c: eb01001f     	cmp	x0, x1
     320: 540008c2     	b.hs	0x438 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x164>
     324: eb03001f     	cmp	x0, x3
     328: 54000c42     	b.hs	0x4b0 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x1dc>
     32c: 91000408     	add	x8, x0, #0x1
     330: eb03011f     	cmp	x8, x3
     334: 54000882     	b.hs	0x444 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x170>
     338: 91000808     	add	x8, x0, #0x2
     33c: eb03011f     	cmp	x8, x3
     340: 54000942     	b.hs	0x468 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x194>
     344: 91000c08     	add	x8, x0, #0x3
     348: eb03011f     	cmp	x8, x3
     34c: 54000a82     	b.hs	0x49c <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x1c8>
     350: bc5fc143     	ldur	s3, [x10, #-0x4]
     354: bc5fc164     	ldur	s4, [x11, #-0x4]
     358: 1e240863     	fmul	s3, s3, s4
     35c: 1e232821     	fadd	s1, s1, s3
     360: bd400143     	ldr	s3, [x10]
     364: bd400164     	ldr	s4, [x11]
     368: 1e240863     	fmul	s3, s3, s4
     36c: 1e232842     	fadd	s2, s2, s3
     370: fc404143     	ldur	d3, [x10, #0x4]
     374: fc404164     	ldur	d4, [x11, #0x4]
     378: 2e24dc63     	fmul.2s	v3, v3, v4
     37c: 0e23d400     	fadd.2s	v0, v0, v3
     380: 91001000     	add	x0, x0, #0x4
     384: 9100414a     	add	x10, x10, #0x10
     388: 9100416b     	add	x11, x11, #0x10
     38c: f1000529     	subs	x9, x9, #0x1
     390: 54fffc61     	b.ne	0x31c <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x48>
     394: 14000023     	b	0x420 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x14c>
     398: d2800000     	mov	x0, #0x0                ; =0
     39c: 9100110a     	add	x10, x8, #0x4
     3a0: 9100104b     	add	x11, x2, #0x4
     3a4: 2f00e400     	movi	d0, #0000000000000000
     3a8: 2f00e401     	movi	d1, #0000000000000000
     3ac: 2f00e402     	movi	d2, #0000000000000000
     3b0: eb01001f     	cmp	x0, x1
     3b4: 54000422     	b.hs	0x438 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x164>
     3b8: 91000408     	add	x8, x0, #0x1
     3bc: eb01011f     	cmp	x8, x1
     3c0: 540004c2     	b.hs	0x458 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x184>
     3c4: 91000808     	add	x8, x0, #0x2
     3c8: eb01011f     	cmp	x8, x1
     3cc: 54000582     	b.hs	0x47c <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x1a8>
     3d0: 91000c08     	add	x8, x0, #0x3
     3d4: eb01011f     	cmp	x8, x1
     3d8: 540005a2     	b.hs	0x48c <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x1b8>
     3dc: bc5fc143     	ldur	s3, [x10, #-0x4]
     3e0: bc5fc164     	ldur	s4, [x11, #-0x4]
     3e4: 1e240863     	fmul	s3, s3, s4
     3e8: 1e232821     	fadd	s1, s1, s3
     3ec: bd400143     	ldr	s3, [x10]
     3f0: bd400164     	ldr	s4, [x11]
     3f4: 1e240863     	fmul	s3, s3, s4
     3f8: 1e232842     	fadd	s2, s2, s3
     3fc: fc404143     	ldur	d3, [x10, #0x4]
     400: fc404164     	ldur	d4, [x11, #0x4]
     404: 2e24dc63     	fmul.2s	v3, v3, v4
     408: 0e23d400     	fadd.2s	v0, v0, v3
     40c: 91001000     	add	x0, x0, #0x4
     410: 9100414a     	add	x10, x10, #0x10
     414: 9100416b     	add	x11, x11, #0x10
     418: f1000529     	subs	x9, x9, #0x1
     41c: 54fffca1     	b.ne	0x3b0 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0xdc>
     420: 1e212841     	fadd	s1, s2, s1
     424: 1e212801     	fadd	s1, s0, s1
     428: 5e0c0400     	mov	s0, v0[1]
     42c: 1e212800     	fadd	s0, s0, s1
     430: a8c17bfd     	ldp	x29, x30, [sp], #0x10
     434: d65f03c0     	ret
     438: 90000002     	adrp	x2, 0x0 <ltmp0>
     43c: 91000042     	add	x2, x2, #0x0
     440: 94000000     	bl	0x440 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x16c>
     444: 90000002     	adrp	x2, 0x0 <ltmp0>
     448: 91000042     	add	x2, x2, #0x0
     44c: aa0803e0     	mov	x0, x8
     450: aa0303e1     	mov	x1, x3
     454: 94000000     	bl	0x454 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x180>
     458: 90000002     	adrp	x2, 0x0 <ltmp0>
     45c: 91000042     	add	x2, x2, #0x0
     460: aa0803e0     	mov	x0, x8
     464: 94000000     	bl	0x464 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x190>
     468: 90000002     	adrp	x2, 0x0 <ltmp0>
     46c: 91000042     	add	x2, x2, #0x0
     470: aa0803e0     	mov	x0, x8
     474: aa0303e1     	mov	x1, x3
     478: 94000000     	bl	0x478 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x1a4>
     47c: 90000002     	adrp	x2, 0x0 <ltmp0>
     480: 91000042     	add	x2, x2, #0x0
     484: aa0803e0     	mov	x0, x8
     488: 94000000     	bl	0x488 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x1b4>
     48c: 90000002     	adrp	x2, 0x0 <ltmp0>
     490: 91000042     	add	x2, x2, #0x0
     494: aa0803e0     	mov	x0, x8
     498: 94000000     	bl	0x498 <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x1c4>
     49c: 90000002     	adrp	x2, 0x0 <ltmp0>
     4a0: 91000042     	add	x2, x2, #0x0
     4a4: aa0803e0     	mov	x0, x8
     4a8: aa0303e1     	mov	x1, x3
     4ac: 94000000     	bl	0x4ac <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x1d8>
     4b0: 90000002     	adrp	x2, 0x0 <ltmp0>
     4b4: 91000042     	add	x2, x2, #0x0
     4b8: aa0303e1     	mov	x1, x3
     4bc: 94000000     	bl	0x4bc <__ZN16dotproduct_bench7dotprod28dot_naive_unroll_no_data_dep17h7567feea3fe63fe2E+0x1e8>

00000000000004c0 <__ZN16dotproduct_bench7dotprod14dot_vectorized17hec15d4c3e5e0caa9E>:
     4c0: a9bf7bfd     	stp	x29, x30, [sp, #-0x10]!
     4c4: 910003fd     	mov	x29, sp
     4c8: d280000a     	mov	x10, #0x0               ; =0
     4cc: d342fc28     	lsr	x8, x1, #2
     4d0: d2e80009     	mov	x9, #0x4000000000000000 ; =4611686018427387904
     4d4: b40001c8     	cbz	x8, 0x50c <__ZN16dotproduct_bench7dotprod14dot_vectorized17hec15d4c3e5e0caa9E+0x4c>
     4d8: d1000529     	sub	x9, x9, #0x1
     4dc: b40001e9     	cbz	x9, 0x518 <__ZN16dotproduct_bench7dotprod14dot_vectorized17hec15d4c3e5e0caa9E+0x58>
     4e0: 91001140     	add	x0, x10, #0x4
     4e4: eb01001f     	cmp	x0, x1
     4e8: 54000228     	b.hi	0x52c <__ZN16dotproduct_bench7dotprod14dot_vectorized17hec15d4c3e5e0caa9E+0x6c>
     4ec: d1000508     	sub	x8, x8, #0x1
     4f0: aa0003ea     	mov	x10, x0
     4f4: eb03001f     	cmp	x0, x3
     4f8: 54fffee9     	b.ls	0x4d4 <__ZN16dotproduct_bench7dotprod14dot_vectorized17hec15d4c3e5e0caa9E+0x14>
     4fc: 90000002     	adrp	x2, 0x0 <ltmp0>
     500: 91000042     	add	x2, x2, #0x0
     504: aa0303e1     	mov	x1, x3
     508: 94000000     	bl	0x508 <__ZN16dotproduct_bench7dotprod14dot_vectorized17hec15d4c3e5e0caa9E+0x48>
     50c: 2f00e400     	movi	d0, #0000000000000000
     510: a8c17bfd     	ldp	x29, x30, [sp], #0x10
     514: d65f03c0     	ret
     518: 90000002     	adrp	x2, 0x0 <ltmp0>
     51c: 91000042     	add	x2, x2, #0x0
     520: 92800060     	mov	x0, #-0x4               ; =-4
     524: d2800001     	mov	x1, #0x0                ; =0
     528: 94000000     	bl	0x528 <__ZN16dotproduct_bench7dotprod14dot_vectorized17hec15d4c3e5e0caa9E+0x68>
     52c: 91001140     	add	x0, x10, #0x4
     530: 90000002     	adrp	x2, 0x0 <ltmp0>
     534: 91000042     	add	x2, x2, #0x0
     538: 94000000     	bl	0x538 <__ZN16dotproduct_bench7dotprod14dot_vectorized17hec15d4c3e5e0caa9E+0x78>

000000000000053c <__ZN16dotproduct_bench7dotprod23dot_vectorized_unrolled17hf436b46e7fca20f1E>:
     53c: a9bf7bfd     	stp	x29, x30, [sp, #-0x10]!
     540: 910003fd     	mov	x29, sp
     544: 2f00e400     	movi	d0, #0000000000000000
     548: f100203f     	cmp	x1, #0x8
     54c: 54000543     	b.lo	0x5f4 <__ZN16dotproduct_bench7dotprod23dot_vectorized_unrolled17hf436b46e7fca20f1E+0xb8>
     550: d343fc29     	lsr	x9, x1, #3
     554: 9100404a     	add	x10, x2, #0x10
     558: 9100400b     	add	x11, x0, #0x10
     55c: 52800100     	mov	w0, #0x8                ; =8
     560: 2f00e401     	movi	d1, #0000000000000000
     564: 2f00e402     	movi	d2, #0000000000000000
     568: d1001008     	sub	x8, x0, #0x4
     56c: eb01011f     	cmp	x8, x1
     570: 54000468     	b.hi	0x5fc <__ZN16dotproduct_bench7dotprod23dot_vectorized_unrolled17hf436b46e7fca20f1E+0xc0>
     574: eb03011f     	cmp	x8, x3
     578: 540004a8     	b.hi	0x60c <__ZN16dotproduct_bench7dotprod23dot_vectorized_unrolled17hf436b46e7fca20f1E+0xd0>
     57c: eb01001f     	cmp	x0, x1
     580: 54000508     	b.hi	0x620 <__ZN16dotproduct_bench7dotprod23dot_vectorized_unrolled17hf436b46e7fca20f1E+0xe4>
     584: eb03001f     	cmp	x0, x3
     588: 54000528     	b.hi	0x62c <__ZN16dotproduct_bench7dotprod23dot_vectorized_unrolled17hf436b46e7fca20f1E+0xf0>
     58c: 3cdf0163     	ldur	q3, [x11, #-0x10]
     590: 3cdf0144     	ldur	q4, [x10, #-0x10]
     594: 3cc20565     	ldr	q5, [x11], #0x20
     598: 3cc20546     	ldr	q6, [x10], #0x20
     59c: 6e24dc63     	fmul.4s	v3, v3, v4
     5a0: 5e1c0464     	mov	s4, v3[3]
     5a4: 5e140467     	mov	s7, v3[2]
     5a8: 5e0c0470     	mov	s16, v3[1]
     5ac: 1e202863     	fadd	s3, s3, s0
     5b0: 1e302863     	fadd	s3, s3, s16
     5b4: 1e272863     	fadd	s3, s3, s7
     5b8: 1e242863     	fadd	s3, s3, s4
     5bc: 1e232821     	fadd	s1, s1, s3
     5c0: 6e26dca3     	fmul.4s	v3, v5, v6
     5c4: 5e1c0464     	mov	s4, v3[3]
     5c8: 5e140465     	mov	s5, v3[2]
     5cc: 5e0c0466     	mov	s6, v3[1]
     5d0: 1e202863     	fadd	s3, s3, s0
     5d4: 1e262863     	fadd	s3, s3, s6
     5d8: 1e252863     	fadd	s3, s3, s5
     5dc: 1e242863     	fadd	s3, s3, s4
     5e0: 1e232842     	fadd	s2, s2, s3
     5e4: 91002000     	add	x0, x0, #0x8
     5e8: f1000529     	subs	x9, x9, #0x1
     5ec: 54fffbe1     	b.ne	0x568 <__ZN16dotproduct_bench7dotprod23dot_vectorized_unrolled17hf436b46e7fca20f1E+0x2c>
     5f0: 1e222820     	fadd	s0, s1, s2
     5f4: a8c17bfd     	ldp	x29, x30, [sp], #0x10
     5f8: d65f03c0     	ret
     5fc: 90000002     	adrp	x2, 0x0 <ltmp0>
     600: 91000042     	add	x2, x2, #0x0
     604: aa0803e0     	mov	x0, x8
     608: 94000000     	bl	0x608 <__ZN16dotproduct_bench7dotprod23dot_vectorized_unrolled17hf436b46e7fca20f1E+0xcc>
     60c: 90000002     	adrp	x2, 0x0 <ltmp0>
     610: 91000042     	add	x2, x2, #0x0
     614: aa0803e0     	mov	x0, x8
     618: aa0303e1     	mov	x1, x3
     61c: 94000000     	bl	0x61c <__ZN16dotproduct_bench7dotprod23dot_vectorized_unrolled17hf436b46e7fca20f1E+0xe0>
     620: 90000002     	adrp	x2, 0x0 <ltmp0>
     624: 91000042     	add	x2, x2, #0x0
     628: 94000000     	bl	0x628 <__ZN16dotproduct_bench7dotprod23dot_vectorized_unrolled17hf436b46e7fca20f1E+0xec>
     62c: 90000002     	adrp	x2, 0x0 <ltmp0>
     630: 91000042     	add	x2, x2, #0x0
     634: aa0303e1     	mov	x1, x3
     638: 94000000     	bl	0x638 <__ZN16dotproduct_bench7dotprod23dot_vectorized_unrolled17hf436b46e7fca20f1E+0xfc>

000000000000063c <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE>:
     63c: a9bf7bfd     	stp	x29, x30, [sp, #-0x10]!
     640: 910003fd     	mov	x29, sp
     644: d342fc29     	lsr	x9, x1, #2
     648: f240043f     	tst	x1, #0x3
     64c: 9a890529     	cinc	x9, x9, ne
     650: b4000229     	cbz	x9, 0x694 <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE+0x58>
     654: aa0003e8     	mov	x8, x0
     658: eb03003f     	cmp	x1, x3
     65c: 54000209     	b.ls	0x69c <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE+0x60>
     660: d2800000     	mov	x0, #0x0                ; =0
     664: 6f00e400     	movi.2d	v0, #0000000000000000
     668: eb01001f     	cmp	x0, x1
     66c: 54000342     	b.hs	0x6d4 <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE+0x98>
     670: eb03001f     	cmp	x0, x3
     674: 54000362     	b.hs	0x6e0 <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE+0xa4>
     678: 3cc10501     	ldr	q1, [x8], #0x10
     67c: 3cc10442     	ldr	q2, [x2], #0x10
     680: 4e21cc40     	fmla.4s	v0, v2, v1
     684: 91001000     	add	x0, x0, #0x4
     688: f1000529     	subs	x9, x9, #0x1
     68c: 54fffee1     	b.ne	0x668 <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE+0x2c>
     690: 1400000d     	b	0x6c4 <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE+0x88>
     694: 6f00e400     	movi.2d	v0, #0000000000000000
     698: 1400000b     	b	0x6c4 <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE+0x88>
     69c: d2800000     	mov	x0, #0x0                ; =0
     6a0: 6f00e400     	movi.2d	v0, #0000000000000000
     6a4: eb01001f     	cmp	x0, x1
     6a8: 54000162     	b.hs	0x6d4 <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE+0x98>
     6ac: 3cc10501     	ldr	q1, [x8], #0x10
     6b0: 3cc10442     	ldr	q2, [x2], #0x10
     6b4: 4e21cc40     	fmla.4s	v0, v2, v1
     6b8: 91001000     	add	x0, x0, #0x4
     6bc: f1000529     	subs	x9, x9, #0x1
     6c0: 54ffff21     	b.ne	0x6a4 <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE+0x68>
     6c4: 6e20d400     	faddp.4s	v0, v0, v0
     6c8: 7e30d800     	faddp.2s	s0, v0
     6cc: a8c17bfd     	ldp	x29, x30, [sp], #0x10
     6d0: d65f03c0     	ret
     6d4: 90000002     	adrp	x2, 0x0 <ltmp0>
     6d8: 91000042     	add	x2, x2, #0x0
     6dc: 94000000     	bl	0x6dc <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE+0xa0>
     6e0: 90000002     	adrp	x2, 0x0 <ltmp0>
     6e4: 91000042     	add	x2, x2, #0x0
     6e8: aa0303e1     	mov	x1, x3
     6ec: 94000000     	bl	0x6ec <__ZN16dotproduct_bench7dotprod19dot_vectorized_neon17h9f30fc28a3a4d4ddE+0xb0>

00000000000006f0 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E>:
     6f0: a9bf7bfd     	stp	x29, x30, [sp, #-0x10]!
     6f4: 910003fd     	mov	x29, sp
     6f8: d343fc29     	lsr	x9, x1, #3
     6fc: f240083f     	tst	x1, #0x7
     700: 9a890529     	cinc	x9, x9, ne
     704: b4000389     	cbz	x9, 0x774 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0x84>
     708: aa0003e8     	mov	x8, x0
     70c: eb03003f     	cmp	x1, x3
     710: 54000389     	b.ls	0x780 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0x90>
     714: d2800000     	mov	x0, #0x0                ; =0
     718: 9100404a     	add	x10, x2, #0x10
     71c: 9100410b     	add	x11, x8, #0x10
     720: 6f00e400     	movi.2d	v0, #0000000000000000
     724: 6f00e401     	movi.2d	v1, #0000000000000000
     728: eb01001f     	cmp	x0, x1
     72c: 540005e2     	b.hs	0x7e8 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0xf8>
     730: eb03001f     	cmp	x0, x3
     734: 54000722     	b.hs	0x818 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0x128>
     738: 91001008     	add	x8, x0, #0x4
     73c: eb01011f     	cmp	x8, x1
     740: 540005a2     	b.hs	0x7f4 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0x104>
     744: eb03011f     	cmp	x8, x3
     748: 540005e2     	b.hs	0x804 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0x114>
     74c: 3cdf0162     	ldur	q2, [x11, #-0x10]
     750: 3cdf0143     	ldur	q3, [x10, #-0x10]
     754: 3cc20564     	ldr	q4, [x11], #0x20
     758: 3cc20545     	ldr	q5, [x10], #0x20
     75c: 4e22cc60     	fmla.4s	v0, v3, v2
     760: 4e24cca1     	fmla.4s	v1, v5, v4
     764: 91002000     	add	x0, x0, #0x8
     768: f1000529     	subs	x9, x9, #0x1
     76c: 54fffde1     	b.ne	0x728 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0x38>
     770: 14000017     	b	0x7cc <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0xdc>
     774: 6f00e401     	movi.2d	v1, #0000000000000000
     778: 6f00e400     	movi.2d	v0, #0000000000000000
     77c: 14000014     	b	0x7cc <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0xdc>
     780: d2800000     	mov	x0, #0x0                ; =0
     784: 9100410a     	add	x10, x8, #0x10
     788: 9100404b     	add	x11, x2, #0x10
     78c: 6f00e400     	movi.2d	v0, #0000000000000000
     790: 6f00e401     	movi.2d	v1, #0000000000000000
     794: eb01001f     	cmp	x0, x1
     798: 54000282     	b.hs	0x7e8 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0xf8>
     79c: 91001008     	add	x8, x0, #0x4
     7a0: eb01011f     	cmp	x8, x1
     7a4: 54000282     	b.hs	0x7f4 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0x104>
     7a8: ad7f9562     	ldp	q2, q5, [x11, #-0x10]
     7ac: ad7f9143     	ldp	q3, q4, [x10, #-0x10]
     7b0: 4e23cc40     	fmla.4s	v0, v2, v3
     7b4: 4e24cca1     	fmla.4s	v1, v5, v4
     7b8: 91002000     	add	x0, x0, #0x8
     7bc: 9100814a     	add	x10, x10, #0x20
     7c0: 9100816b     	add	x11, x11, #0x20
     7c4: f1000529     	subs	x9, x9, #0x1
     7c8: 54fffe61     	b.ne	0x794 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0xa4>
     7cc: 6e20d400     	faddp.4s	v0, v0, v0
     7d0: 7e30d800     	faddp.2s	s0, v0
     7d4: 6e21d421     	faddp.4s	v1, v1, v1
     7d8: 7e30d821     	faddp.2s	s1, v1
     7dc: 1e212800     	fadd	s0, s0, s1
     7e0: a8c17bfd     	ldp	x29, x30, [sp], #0x10
     7e4: d65f03c0     	ret
     7e8: 90000002     	adrp	x2, 0x0 <ltmp0>
     7ec: 91000042     	add	x2, x2, #0x0
     7f0: 94000000     	bl	0x7f0 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0x100>
     7f4: 90000002     	adrp	x2, 0x0 <ltmp0>
     7f8: 91000042     	add	x2, x2, #0x0
     7fc: aa0803e0     	mov	x0, x8
     800: 94000000     	bl	0x800 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0x110>
     804: 91001000     	add	x0, x0, #0x4
     808: 90000002     	adrp	x2, 0x0 <ltmp0>
     80c: 91000042     	add	x2, x2, #0x0
     810: aa0303e1     	mov	x1, x3
     814: 94000000     	bl	0x814 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0x124>
     818: 90000002     	adrp	x2, 0x0 <ltmp0>
     81c: 91000042     	add	x2, x2, #0x0
     820: aa0303e1     	mov	x1, x3
     824: 94000000     	bl	0x824 <__ZN16dotproduct_bench7dotprod28dot_vectorized_neon_unrolled17hef4638c55da80600E+0x134>
```
