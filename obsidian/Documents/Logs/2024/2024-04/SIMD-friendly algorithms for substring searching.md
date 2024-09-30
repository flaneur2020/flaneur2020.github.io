常见的 strstr 的算法可以分为两类：1. 确定性的自动机，比如 KMP、Boyer Moore 算法；2. 基于一个简单的对比，像 Karp Rabin 算法。

这些算法的问题是，它们假设查一个额外的表和判断是廉价的，而比较 string 本身是昂贵的。

但是现代的 CPU 不满足上述假设：

1. 在 64 bit CPU 上对比 1、2、4、8 个 bytes 没有区别，如果支持 SIMD，那么对比 32 甚至 64 个 bytes，和比较一个 byte 的成本是一样的
2. 因此简单的比较 short sequence 的算法，得比 fancy 的避免如此比较的算法更快
3. 检查一个表，涉及内存访问，至少有一次 L1 cache 的开销（3 cycles），读 char 的访存开销也一样；
4. 错误的分支预测代缴比较高（10～20 cycles）
5. 有一些隐藏的依赖关系：1. read char；2. 比较；3. 判断跳转；难以应用乱序执行能力

## Solution

Karp-Rabin 算法的思路：计算一下 substring 的 hash 值，然后计按窗口计算哈希值。每次迭代时，第二个哈希值的修改代价很小。

```
k  := substring length
h1 := hash(substring)
h2 := hash(string[i .. i + k])
for i in 0 .. n - k loop
    if h1 == h2 then
        if substring == string[i .. i + k] then
            return i
        end if
    end if

    h = next_hash(h, ...) # this meant to be cheap
end loop
```

SIMD 的解决方法就是，将 hash predicate 向量化，并行计算 N 个哈希值。

另一个层面的优化是，在哈希值相等之后的 memcmp 判断，因为已知 substring 的长度，因此可以针对不同的长度来编写不同的特化实现。
## Algorithm 1: Generic SIMD

它比较 substring 的第一个和最后一个 bytes（比如叫做 F 和 L），用作剪枝。

两个 substring 的 F 和 L 都相同，再做后续的 memcmp 比较。
## Implementation

### SSE & AVX2

```C++
size_t avx2_strstr_anysize(const char* s, size_t n, const char* needle, size_t k) {

    const __m256i first = _mm256_set1_epi8(needle[0]);
    const __m256i last  = _mm256_set1_epi8(needle[k - 1]);

    for (size_t i = 0; i < n; i += 32) {

        const __m256i block_first = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
        const __m256i block_last  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + k - 1));

        const __m256i eq_first = _mm256_cmpeq_epi8(first, block_first);
        const __m256i eq_last  = _mm256_cmpeq_epi8(last, block_last);

        uint32_t mask = _mm256_movemask_epi8(_mm256_and_si256(eq_first, eq_last));

        while (mask != 0) {

            const auto bitpos = bits::get_first_bit_set(mask);

            if (memcmp(s + i + bitpos + 1, needle + 1, k - 2) == 0) {
                return i + bitpos;
            }

            mask = bits::clear_leftmost_set(mask);
        }
    }

    return std::string::npos;
```

## SWAR

SWAR 好像是 “SIMD Within a Register” 的意思。好像思路是通过 uint64_t 一次性计算 8 个 char 的技术。

```C++
size_t swar64_strstr_anysize(const char* s, size_t n, const char* needle, size_t k) {

    const uint64_t first = 0x0101010101010101llu * static_cast<uint8_t>(needle[0]);
    const uint64_t last  = 0x0101010101010101llu * static_cast<uint8_t>(needle[k - 1]);

    uint64_t* block_first = reinterpret_cast<uint64_t*>(const_cast<char*>(s));
    uint64_t* block_last  = reinterpret_cast<uint64_t*>(const_cast<char*>(s + k - 1));

    for (auto i=0u; i < n; i+=8, block_first++, block_last++) {
        const uint64_t eq = (*block_first ^ first) | (*block_last ^ last);

        const uint64_t t0 = (~eq & 0x7f7f7f7f7f7f7f7fllu) + 0x0101010101010101llu;
        const uint64_t t1 = (~eq & 0x8080808080808080llu);
        uint64_t zeros = t0 & t1;
        size_t j = 0;

        while (zeros) {
            if (zeros & 0x80) {
                const char* substr = reinterpret_cast<char*>(block_first) + j + 1;
                if (memcmp(substr, needle + 1, k - 2) == 0) {
                    return i + j;
                }
            }

            zeros >>= 8;
            j += 1;
        }
    }

    return std::string::npos;
}
```
## ARM Neon (32 bit code)

```C++
size_t FORCE_INLINE neon_strstr_anysize(const char* s, size_t n, const char* needle, size_t k) {

    assert(k > 0);
    assert(n > 0);

    const uint8x16_t first = vdupq_n_u8(needle[0]);
    const uint8x16_t last  = vdupq_n_u8(needle[k - 1]);
    const uint8x8_t  half  = vdup_n_u8(0x0f);

    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(s);

    union {
        uint8_t  tmp[8];
        uint32_t word[2];
    };

    for (size_t i = 0; i < n; i += 16) {

        const uint8x16_t block_first = vld1q_u8(ptr + i);
        const uint8x16_t block_last  = vld1q_u8(ptr + i + k - 1);

        const uint8x16_t eq_first = vceqq_u8(first, block_first);
        const uint8x16_t eq_last  = vceqq_u8(last, block_last);
        const uint8x16_t pred_16  = vandq_u8(eq_first, eq_last);
        const uint8x8_t pred_8    = vbsl_u8(half, vget_low_u8(pred_16), vget_high_u8(pred_16));

        vst1_u8(tmp, pred_8);

        if ((word[0] | word[1]) == 0) {
            continue;
        }

        for (int j=0; j < 8; j++) {
            if (tmp[j] & 0x0f) {
                if (memcmp(s + i + j + 1, needle + 1, k - 2) == 0) {
                    return i + j;
                }
            }
        }

        for (int j=0; j < 8; j++) {
            if (tmp[j] & 0xf0) {
                if (memcmp(s + i + j + 1 + 8, needle + 1, k - 2) == 0) {
                    return i + j + 8;
                }
            }
        }
    }

    return std::string::npos;
}
```