# chicago-chase

Chicago Chase - Quantum Star LLC

Quantum-inspired classical optimization primitives, starting with simulated
annealing. The engine-plane artifact is the importable
`pkg/quantumstar/optimizer` package; `cmd/chicago-chase` is a runnable demo
that uses it to find a minimum-variance portfolio over a small set of assets.

## Usage

```sh
go run ./cmd/chicago-chase   # run the portfolio demo
go test ./...                # run tests
go build ./...                # build
```

Or via the Makefile: `make run`, `make test`, `make build`.
