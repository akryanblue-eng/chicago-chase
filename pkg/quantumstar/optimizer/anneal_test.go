package optimizer

import (
	"math"
	"testing"
)

func sphere(x []float64) float64 {
	sum := 0.0
	for _, v := range x {
		sum += v * v
	}
	return sum
}

func TestAnneal_ConvergesToSphereMinimum(t *testing.T) {
	bounds := Bounds{Min: []float64{-10, -10}, Max: []float64{10, 10}}
	cfg := Config{
		InitialTemp:   10,
		CoolingRate:   0.995,
		MaxIterations: 2000,
		Seed:          42,
	}

	result := Anneal(sphere, []float64{8, -7}, bounds, cfg)

	if result.Cost > 0.05 {
		t.Fatalf("expected cost near 0, got %f with solution %v", result.Cost, result.Solution)
	}
	for _, v := range result.Solution {
		if math.Abs(v) > 0.5 {
			t.Fatalf("expected solution near origin, got %v", result.Solution)
		}
	}
}

func TestAnneal_DeterministicForFixedSeed(t *testing.T) {
	bounds := Bounds{Min: []float64{-10}, Max: []float64{10}}
	cfg := Config{InitialTemp: 5, CoolingRate: 0.99, MaxIterations: 500, Seed: 7}

	r1 := Anneal(sphere, []float64{6}, bounds, cfg)
	r2 := Anneal(sphere, []float64{6}, bounds, cfg)

	if r1.Cost != r2.Cost || r1.Solution[0] != r2.Solution[0] {
		t.Fatalf("expected identical results for the same seed, got %+v vs %+v", r1, r2)
	}
}
