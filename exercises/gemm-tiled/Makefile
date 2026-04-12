BINARY := gemm-tiled
CSV ?= benchmark.csv
SVG ?= benchmark.svg
PROBLEMS ?= 1024,1280,1536,1792,2048
ITERATIONS ?= 10
WARMUP ?= 2
IMPLEMENTATIONS ?= veclib,mps,metal-best

.PHONY: build benchmark quick plot clean

build:
	swift build

benchmark:
	swift run -c release $(BINARY) \
	  --problems $(PROBLEMS) \
	  --iterations $(ITERATIONS) \
	  --warmup $(WARMUP) \
	  --implementations "$(IMPLEMENTATIONS)" \
	  --csv $(CSV)
	python3 scripts/plot_benchmark.py $(CSV) $(SVG)

quick:
	swift run -c release $(BINARY) \
	  --problems 64,128,256 \
	  --iterations 1 \
	  --warmup 1 \
	  --csv $(CSV)

plot:
	python3 scripts/plot_benchmark.py $(CSV) $(SVG)

clean:
	swift package clean
