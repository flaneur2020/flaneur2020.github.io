BINARY := gemm-tiled
CSV ?= benchmark.csv
SVG ?= benchmark.svg
PROBLEMS ?= 128,192,256,384,512,640,768,896,1024,1152,1280
ITERATIONS ?= 10
WARMUP ?= 2

.PHONY: build benchmark quick plot clean

build:
	swift build

benchmark:
	swift run $(BINARY) \
	  --problems $(PROBLEMS) \
	  --iterations $(ITERATIONS) \
	  --warmup $(WARMUP) \
	  --csv $(CSV)
	python3 scripts/plot_benchmark.py $(CSV) $(SVG)

quick:
	swift run $(BINARY) \
	  --problems 64,128,256 \
	  --iterations 1 \
	  --warmup 1 \
	  --csv $(CSV)

plot:
	python3 scripts/plot_benchmark.py $(CSV) $(SVG)

clean:
	swift package clean
