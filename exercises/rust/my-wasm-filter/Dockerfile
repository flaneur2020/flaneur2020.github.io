FROM istio/proxyv2:1.6.0
ADD envoy.yaml /etc/envoy.yaml
ADD target/wasm32-unknown-unknown/release/my_wasm_filter.wasm /usr/local/lib/my_wasm_filter.wasm
ENTRYPOINT /usr/local/bin/envoy -c /etc/envoy.yaml -l info --service-cluster proxy
