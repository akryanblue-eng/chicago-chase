package optimizer

import (
	"math"
	"math/rand"
)

// Anneal performs simulated annealing over objective within bounds, starting
// from start, and returns the best solution found. The schedule and random
// source are fully determined by cfg.Seed, making runs reproducible.
func Anneal(objective ObjectiveFunc, start []float64, bounds Bounds, cfg Config) Result {
	rng := rand.New(rand.NewSource(cfg.Seed))

	current := append([]float64{}, start...)
	currentCost := objective(current)

	best := append([]float64{}, current...)
	bestCost := currentCost

	temp := cfg.InitialTemp

	for i := 0; i < cfg.MaxIterations; i++ {
		candidate := neighbor(current, bounds, temp, rng)
		candidateCost := objective(candidate)

		if accept(currentCost, candidateCost, temp, rng) {
			current = candidate
			currentCost = candidateCost

			if currentCost < bestCost {
				best = append([]float64{}, current...)
				bestCost = currentCost
			}
		}

		temp *= cfg.CoolingRate
	}

	return Result{Solution: best, Cost: bestCost, Iterations: cfg.MaxIterations}
}

func neighbor(x []float64, bounds Bounds, temp float64, rng *rand.Rand) []float64 {
	next := make([]float64, len(x))
	for i, v := range x {
		step := (rng.Float64()*2 - 1) * temp
		nv := v + step
		if nv < bounds.Min[i] {
			nv = bounds.Min[i]
		}
		if nv > bounds.Max[i] {
			nv = bounds.Max[i]
		}
		next[i] = nv
	}
	return next
}

func accept(currentCost, candidateCost, temp float64, rng *rand.Rand) bool {
	if candidateCost < currentCost {
		return true
	}
	if temp <= 0 {
		return false
	}
	delta := candidateCost - currentCost
	return rng.Float64() < math.Exp(-delta/temp)
}
