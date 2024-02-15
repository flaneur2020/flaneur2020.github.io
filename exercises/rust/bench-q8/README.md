```
test tests::bench_vec_dot_q8_ggml                     ... bench:         703 ns/iter (+/- 16)
test tests::bench_vec_dot_q8_naive                    ... bench:      10,834 ns/iter (+/- 190)
test tests::bench_vec_dot_q8_neon                     ... bench:         955 ns/iter (+/- 21)
test tests::bench_vec_dot_q8_neon_unrolled            ... bench:         712 ns/iter (+/- 4)
test tests::bench_vec_dot_q8_neon_unrolled_single_sum ... bench:         953 ns/iter (+/- 17)
test tests::bench_vec_dot_q8_neon_unrolled_vfma       ... bench:         702 ns/iter (+/- 8)
test tests::bench_vec_dot_q8_stdsimd                  ... bench:       3,063 ns/iter (+/- 19)
```