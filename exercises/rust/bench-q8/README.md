```
test tests::bench_vec_dot_q8_ggml                     ... bench:       1,611 ns/iter (+/- 27)
test tests::bench_vec_dot_q8_naive                    ... bench:      21,872 ns/iter (+/- 184)
test tests::bench_vec_dot_q8_neon                     ... bench:       1,998 ns/iter (+/- 14)
test tests::bench_vec_dot_q8_neon_unrolled            ... bench:       1,613 ns/iter (+/- 8)
test tests::bench_vec_dot_q8_neon_unrolled_single_sum ... bench:       1,998 ns/iter (+/- 18)
test tests::bench_vec_dot_q8_neon_unrolled_vfma       ... bench:       1,603 ns/iter (+/- 8)
test tests::bench_vec_dot_q8_stdsimd                  ... bench:       6,305 ns/iter (+/- 58)
```