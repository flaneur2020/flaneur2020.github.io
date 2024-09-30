
TLDR:

- 里面的量化的办法有点奇怪；
- 用 goasm 做手写 SIMD + 量化 有最好的性能；
- goasm 没有 cgo 的开销；
- 可以软件探测一下如果不支持 avx512，则回退回 go 版本；

---
## Quantization

作者的 vector 有 1536 个维度，按 f32 来算，每个 vector 有 6kb。

```
func DotInt8BCE(a, b []int8) int32 {
	if len(a) != len(b) {
		panic("slices must have equal lengths")
	}
 
	sum := int32(0)
	for i := 0; i < len(a); i += 4 {
		aTmp := a[i : i+4 : i+4]
		bTmp := b[i : i+4 : i+4]
		s0 := int32(aTmp[0]) * int32(bTmp[0])
		s1 := int32(aTmp[1]) * int32(bTmp[1])
		s2 := int32(aTmp[2]) * int32(bTmp[2])
		s3 := int32(aTmp[3]) * int32(bTmp[3])
		sum += s0 + s1 + s2 + s3
	}
	return sum
}
```

## SIMD

```
#include "textflag.h"
TEXT ·DotAVX2(SB), NOSPLIT, $0-52
	// Offsets based on slice header offsets.
	// To check, use `GOARCH=amd64 go vet`
	MOVQ a_base+0(FP), AX
	MOVQ b_base+24(FP), BX
	MOVQ a_len+8(FP), DX
	XORQ R8, R8 // return sum
	// Zero Y0, which will store 8 packed 32-bit sums
	VPXOR Y0, Y0, Y0
// In blockloop, we calculate the dot product 16 at a time
blockloop:
	CMPQ DX, $16
	JB reduce
	// Sign-extend 16 bytes into 16 int16s
	VPMOVSXBW (AX), Y1
	VPMOVSXBW (BX), Y2
	// Multiply words vertically to form doubleword intermediates,
	// then add adjacent doublewords.
	VPMADDWD Y1, Y2, Y1
	// Add results to the running sum
	VPADDD Y0, Y1, Y0
	ADDQ $16, AX
	ADDQ $16, BX
	SUBQ $16, DX
	JMP blockloop
reduce:
	// X0 is the low bits of Y0.
	// Extract the high bits into X1, fold in half, add, repeat.
	VEXTRACTI128 $1, Y0, X1
	VPADDD X0, X1, X0
	VPSRLDQ $8, X0, X1
	VPADDD X0, X1, X0
	VPSRLDQ $4, X0, X1
	VPADDD X0, X1, X0
	// Store the reduced sum
	VMOVD X0, R8
end:
	MOVL R8, ret+48(FP)
	VZEROALL
	RET
```

- [`VPMOVSXBW`](https://www.felixcloutier.com/x86/pmovsx), which loads `int8`s into a vector `int16`s
- [`VPMADDWD`](https://www.felixcloutier.com/x86/pmaddwd), which multiplies two `int16` vectors element-wise, then adds fuzzy stack. together adjacent pairs to produce a vector of `int32`s
- [`VPADDD`](https://www.felixcloutier.com/x86/paddb:paddw:paddd:paddq), which accumulates the resulting `int32` vector into our running sum

go 中可以使用 custom aseembler，没有 cgo 的开销。

如果不支持 avx512，可以回退为较慢的 go 版本。

最后的性能表：

- `DotNaive` 0.94M vec/s
- `DotUnroll4` 1.3M vec/s
- `DotBCE` 1.4M vec/s
- `DotInt8BCE` 1.2M vec/s
- `DotAVX2` 7.0M vec/s
- `DotVNNI` 8.8M vec/s

