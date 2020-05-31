build:
	cargo +nightly build --target=wasm32-unknown-unknown --release
	cp target/wasm32-unknown-unknown/release/my_wasm_filter.wasm ./my_wasm_filter.wasm
	docker build . -t my-wasm-filter

run:
	docker run -ti \
		-p 7001:7001 -p 7002:7002 \
		my-wasm-filter:latest \
		/usr/local/bin/envoy -c /etc/envoy.yaml -l debug --service-cluster proxy
