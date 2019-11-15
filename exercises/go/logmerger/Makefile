build:
	go mod tidy
	go build -o bin/logmerger ./cmd/logmerger/main.go


test:
	go test -coverprofile cover.out ./...
	go tool cover -html=cover.out -o cover.html
