# gemm-tiled

This scaffold benchmarks a basic tiled GEMM written in Apple's Metal against Apple vecLib on macOS Apple silicon.

## What it includes

- A baseline `vecLib` GEMM using `Accelerate` + `cblas_sgemm`
- A basic `16x16` tiled Metal compute kernel
- Console output with `MNK` on the X-axis and `MFLOPs` on the Y-axis
- CSV export for plotting performance curves

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
  --csv benchmark.csv
```

Or disable CSV output:

```bash
swift run gemm-tiled --no-csv
```

## CSV columns

- `mnk`: the X-axis value, computed as `M * N * K`
- `mflops`: the Y-axis value, computed from `2 * M * N * K / time`
- `average_ms`: average runtime across measured iterations
- `best_ms`: fastest measured runtime
- `max_abs_error`: maximum absolute difference against the vecLib result

## Why vecLib

`cblas_sgemm` in `Accelerate` is the native Apple baseline that is available on M1/M2/M3 Macs through vecLib, so it is a more relevant comparison than MKL on macOS.

## Where to extend next

- Add a swizzled or block-packed Metal kernel
- Tune tile sizes per GPU family
- Split host-to-device transfer time from pure kernel time
- Add a plotting script for `benchmark.csv`
