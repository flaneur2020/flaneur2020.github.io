# gemm-tiled

This scaffold benchmarks a basic tiled GEMM written in Apple's Metal against Apple vecLib on macOS Apple silicon.

## What it includes

- A baseline `vecLib` GEMM using `Accelerate` + `cblas_sgemm`
- Basic Metal compute kernels in tiled `16x16`, tiled `32x32`, swizzled `32x32`, and register-blocked `4x4` variants, including packed/swizzled-`B`, packed/vectorized-`B`, packed+swizzled-vec4-`B`, packed-vectorized-`B` `k16`, packed-vectorized-`A+B` `k16`, packed-vectorized-`A+B` aligned-only, a `storageModePrivate` aligned variant, aligned tile/threadgroup autotune variants (`64x32x16`, `32x64x16`, `32x32x32`), plus `64x32x16` unrolled and pipelined experiments, and packed-vectorized-`A+B` unrolled-inner-`K` versions
- Console output with `MNK` on the X-axis and `MFLOPs` on the Y-axis
- A `metal-best` autotuned runner for focused `vecLib` vs strongest-Metal comparisons
- CSV export for plotting performance curves, with unified wall-time columns and optional Metal GPU timestamp columns
- A dependency-free SVG plotting script at `scripts/plot_benchmark.py`
- A small `Makefile` for build, benchmark, and plot commands

## Run it

```bash
swift run gemm-tiled
```

That writes `benchmark.csv` by default.

You can also customize the sweep:

```bash
swift run gemm-tiled \
  --problems 128,256,384,512,768,1024 \
  --iterations 10 \
  --warmup 2 \
  --implementations veclib,metal-best \
  --csv benchmark.csv
```

Or disable CSV output:

```bash
swift run gemm-tiled --no-csv
```

The default benchmark compares:

- `vecLib cblas_sgemm`
- `Metal tiled 16x16`
- `Metal tiled 32x32`
- `Metal swizzled 32x32`
- `Metal register blocked 4x4`
- `Metal packed-swizzled B 4x4`
- `Metal packed-vectorized B 4x4`
- `Metal packed-swizzled vec4 B 4x4`
- `Metal packed-vectorized B 4x4 k16`
- `Metal packed-vectorized A+B 4x4 k16`
- `Metal packed-vectorized A+B 4x4 aligned`
- `Metal packed-vectorized A+B 4x4 aligned private`
- `Metal packed-vectorized A+B 64x32x16`
- `Metal packed-vectorized A+B 64x32x16 unroll`
- `Metal packed A+B 64x32x16 pipe`
- `Metal packed-vectorized A+B 32x64x16`
- `Metal packed-vectorized A+B 32x32x32`
- `Metal packed A+B aligned pipe`
- `Metal packed-vectorized A+B 4x4 unroll`

## Make targets

The `Mekfile` is not needed here; plain `make` already picks up `Makefile`.

```bash
make benchmark
make quick
make plot
```

`make benchmark` now runs a focused `vecLib` vs `metal-best` release benchmark and generates the SVG graph in one step.

Useful overrides:

```bash
make benchmark PROBLEMS=128,256,512 ITERATIONS=5 WARMUP=1 IMPLEMENTATIONS=all CSV=custom.csv SVG=custom.svg
make plot CSV=custom.csv SVG=custom.svg
```

Available targets:

- `make build`
- `make benchmark`
- `make benchmark IMPLEMENTATIONS=all`
- `make quick`
- `make plot`
- `make clean`

## Plot a graph

Run the benchmark and then render the SVG chart:

```bash
make benchmark
```

Or rerender an existing CSV:

```bash
make plot CSV=benchmark.csv SVG=benchmark.svg
```

The generated chart uses:

- `MNK` on the X-axis
- `MFLOPs` on the Y-axis, computed from unified wall time
- By default, one line for `vecLib` and one line for `metal-best`
- Override `IMPLEMENTATIONS=all` to plot the full kernel sweep

## CSV columns

- `mnk`: the X-axis value, computed as `M * N * K`
- `average_ms`: average wall-clock runtime across measured iterations
- `best_ms`: fastest wall-clock runtime across measured iterations
- `mflops`: wall-time MFLOPs, computed from `2 * M * N * K / average_ms`
- `device_average_ms`: Metal GPU timestamp average when available
- `device_best_ms`: Metal GPU timestamp best when available
- `device_mflops`: MFLOPs computed from `device_average_ms` when available
- `selected_variant`: when using `metal-best`, the concrete Metal kernel chosen for that problem
- `max_abs_error`: maximum absolute difference against the vecLib result

## Why vecLib

`cblas_sgemm` in `Accelerate` is the native Apple baseline that is available on M1/M2/M3 Macs through vecLib, so it is a more relevant comparison than MKL on macOS. The default table, CSV, and SVG now use unified wall time so vecLib and Metal are compared on the same timing basis.

## Optimization takeaways

- `4x4` register blocking is the first big step up from naive tiled kernels because each thread reuses loaded values across a small output patch instead of producing a single scalar.
- Packing only `B` helps mostly by cleaning up global-memory access, but packing both `A` and `B` is the bigger jump because it lines up both operands with the kernel's `float4` access pattern.
- The aligned-only kernels matter because they remove edge handling from the hot path and make the `k16` packed path more stable on large square problems.
- Tile and threadgroup shape tuning is still one of the highest-value levers: the current hot path has shifted toward the `64x64` family, especially once the benchmark is narrowed to the strongest candidates.
- A focused `metal-best` runner is more trustworthy than sweeping every experiment in one pass, because thermal drift and benchmark order can swamp the small differences between the top kernels.
- Swizzle, pipeline, and manual unroll changes are more situational here; they are useful experiments, but so far they have not beaten the simpler packed `A+B` hot paths as consistently as packing plus tile autotuning.

## Where to extend next

- Add a swizzled or block-packed Metal kernel
- Tune tile sizes per GPU family
- Push `storageModePrivate` + staging onto more kernels
- Emit multiple chart variants such as best-ms or GFLOPs
