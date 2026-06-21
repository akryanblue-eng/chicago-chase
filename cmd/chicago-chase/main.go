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
	variance := rawVariance(weights)

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

func rawVariance(weights []float64) float64 {
	var variance float64
	for i := range weights {
		for j := range weights {
			variance += weights[i] * weights[j] * covariance[i][j]
		}
	}
	return variance
}

func riskContributions(weights []float64) []float64 {
	rc := make([]float64, len(weights))
	for i := range covariance {
		var mrc float64
		for j, wj := range weights {
			mrc += covariance[i][j] * wj
		}
		rc[i] = weights[i] * mrc
	}
	return rc
}

func bounds() optimizer.Bounds {
	return optimizer.Bounds{
		Min: []float64{0, 0, 0, 0},
		Max: []float64{1, 1, 1, 1},
	}
}

func config(seed int64) optimizer.Config {
	return optimizer.Config{
		InitialTemp:   1.0,
		CoolingRate:   0.999,
		MaxIterations: 20000,
		Seed:          seed,
	}
}

// printRiskBlock prints weights, raw variance, per-asset risk contribution,
// and the max risk-contribution deviation for a candidate portfolio.
func printRiskBlock(weights []float64) {
	rc := riskContributions(weights)
	maxRC, minRC := rc[0], rc[0]
	for _, v := range rc {
		if v > maxRC {
			maxRC = v
		}
		if v < minRC {
			minRC = v
		}
	}

	fmt.Printf("weights: %.4f\n", weights)
	fmt.Printf("variance: %.6f\n", rawVariance(weights))
	fmt.Println("RC:")
	for i, v := range rc {
		fmt.Printf("  asset%d: %.6f\n", i, v)
	}
	fmt.Printf("max deviation: %.6f\n", maxRC-minRC)
}

func main() {
	start := []float64{0.25, 0.25, 0.25, 0.25}

	fmt.Println("=== MIN VARIANCE ===")
	minVar := optimizer.Anneal(portfolioVariance, start, bounds(), config(1))
	fmt.Printf("weights: %.4f\n", minVar.Solution)
	fmt.Printf("variance: %.6f\n", minVar.Cost)

	fmt.Println()
	fmt.Println("=== RISK PARITY ===")
	riskParity := optimizer.RiskParityObjective(covariance, 10, 100, 10)
	rp := optimizer.Anneal(riskParity, start, bounds(), config(1))
	printRiskBlock(rp.Solution)

	fmt.Println()
	fmt.Println("=== CURRICULUM (min-variance -> risk parity at 30%) ===")
	scheduler := optimizer.LinearCurriculum(portfolioVariance, riskParity, 0.3)
	curriculum := optimizer.RunCurriculum(scheduler, 0.3, start, bounds(), config(1), 20000)
	printRiskBlock(curriculum.Solution)
}
