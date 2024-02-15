```
test tests::bench_vec_dot_q8_ggml          ... bench:       1,606 ns/iter (+/- 713)
test tests::bench_vec_dot_q8_naive         ... bench:      22,291 ns/iter (+/- 656)
test tests::bench_vec_dot_q8_neon          ... bench:       1,998 ns/iter (+/- 53)
test tests::bench_vec_dot_q8_neon_unrolled ... bench:       1,613 ns/iter (+/- 6)
test tests::bench_vec_dot_q8_stdsimd       ... bench:       6,302 ns/iter (+/- 107)
```