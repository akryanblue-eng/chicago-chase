.PHONY: build test run tidy

build:
	go build ./...

test:
	go test ./...

run:
	go run ./cmd/chicago-chase

tidy:
	go mod tidy
