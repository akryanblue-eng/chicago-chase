// Package optimizer provides quantum-inspired classical optimization
// primitives (starting with simulated annealing) for continuous,
// box-constrained problems.
package optimizer

// ObjectiveFunc evaluates the cost of a candidate solution. Lower is better.
type ObjectiveFunc func(x []float64) float64

// Bounds defines the inclusive search range for each dimension.
type Bounds struct {
	Min []float64
	Max []float64
}

// Config controls the simulated annealing schedule.
type Config struct {
	InitialTemp   float64
	CoolingRate   float64 // multiplicative decay per iteration, in (0, 1)
	MaxIterations int
	Seed          int64
}

// Result holds the best solution found during a run.
type Result struct {
	Solution   []float64
	Cost       float64
	Iterations int
}
