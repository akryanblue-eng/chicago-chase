package main

import (
	"fmt"

	"github.com/akryanblue-eng/chicago-chase/pkg/quantumstar/optimizer"
)

// covariance is an illustrative annualized covariance matrix for 4 assets.
var covariance = [][]float64{
	{0.040, 0.010, 0.004, 0.002},
	{0.010, 0.030, 0.006, 0.003},
	{0.004, 0.006, 0.020, 0.001},
	{0.002, 0.003, 0.001, 0.015},
}

// portfolioVariance is the objective for a long-only, fully-invested
// minimum-variance portfolio: w^T * Sigma * w, with penalties enforcing
// sum(w) == 1 and w >= 0.
func portfolioVariance(weights []float64) float64 {
	var variance float64
	for i := range weights {
		for j := range weights {
			variance += weights[i] * weights[j] * covariance[i][j]
		}
	}

	sum := 0.0
	for _, w := range weights {
		sum += w
		if w < 0 {
			variance += 10 * w * w
		}
	}
	variance += 10 * (sum - 1) * (sum - 1)

	return variance
}

func main() {
	bounds := optimizer.Bounds{
		Min: []float64{0, 0, 0, 0},
		Max: []float64{1, 1, 1, 1},
	}
	cfg := optimizer.Config{
		InitialTemp:   1.0,
		CoolingRate:   0.999,
		MaxIterations: 20000,
		Seed:          1,
	}

	start := []float64{0.25, 0.25, 0.25, 0.25}
	result := optimizer.Anneal(portfolioVariance, start, bounds, cfg)

	fmt.Println("Chicago Chase — minimum-variance portfolio demo")
	fmt.Printf("Weights: %.4f\n", result.Solution)
	fmt.Printf("Portfolio variance: %.6f\n", result.Cost)
}
