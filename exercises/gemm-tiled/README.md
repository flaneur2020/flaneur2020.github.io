# gemm-tiled

This scaffold benchmarks a basic tiled GEMM written in Apple's Metal against Apple vecLib on macOS Apple silicon.

## What it includes

- A baseline `vecLib` GEMM using `Accelerate` + `cblas_sgemm`
- Basic Metal compute kernels in tiled `16x16`, tiled `32x32`, swizzled `32x32`, and register-blocked `4x4` variants, including a packed/swizzled-`B` version
- Console output with `MNK` on the X-axis and `MFLOPs` on the Y-axis
- CSV export for plotting performance curves
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

## Make targets

The `Mekfile` is not needed here; plain `make` already picks up `Makefile`.

```bash
make benchmark
make quick
make plot
```

`make benchmark` now runs the benchmark and generates the SVG graph in one step.

Useful overrides:

```bash
make benchmark PROBLEMS=128,256,512 ITERATIONS=5 WARMUP=1 CSV=custom.csv SVG=custom.svg
make plot CSV=custom.csv SVG=custom.svg
```

Available targets:

- `make build`
- `make benchmark`
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
- `MFLOPs` on the Y-axis
- One line per implementation

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
- Emit multiple chart variants such as best-ms or GFLOPs
