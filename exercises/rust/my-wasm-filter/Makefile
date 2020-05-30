build:
	cargo +nightly build --target=wasm32-unknown-unknown --release
	docker build . -t my-wasm-filter

run:
	docker run -ti \
		--network=host -p 7001 -p 7002 \
		my-wasm-filter:latest \
		/usr/local/bin/envoy -c /etc/envoy.yaml -l debug --service-cluster proxy
