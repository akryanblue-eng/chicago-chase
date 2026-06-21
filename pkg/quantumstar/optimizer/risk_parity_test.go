package optimizer

import "testing"

func TestRiskParityObjective_ConvergesToEqualRiskContribution(t *testing.T) {
	cov := [][]float64{
		{0.040, 0.010, 0.004, 0.002},
		{0.010, 0.030, 0.006, 0.003},
		{0.004, 0.006, 0.020, 0.001},
		{0.002, 0.003, 0.001, 0.015},
	}

	objective := RiskParityObjective(cov, 10, 100, 10)

	bounds := Bounds{
		Min: []float64{0, 0, 0, 0},
		Max: []float64{1, 1, 1, 1},
	}
	cfg := Config{
		InitialTemp:   1.0,
		CoolingRate:   0.999,
		MaxIterations: 20000,
		Seed:          1,
	}

	result := Anneal(objective, []float64{0.25, 0.25, 0.25, 0.25}, bounds, cfg)
	w := result.Solution

	var sum float64
	for _, wi := range w {
		if wi < -1e-4 {
			t.Fatalf("expected long-only weights, got %v", w)
		}
		sum += wi
	}
	if d := sum - 1; d > 1e-3 || d < -1e-3 {
		t.Fatalf("expected sum(w) close to 1, got %f (weights %v)", sum, w)
	}

	mrc := matVec(cov, w)
	maxRC, minRC := w[0]*mrc[0], w[0]*mrc[0]
	for i, wi := range w {
		rc := wi * mrc[i]
		if rc > maxRC {
			maxRC = rc
		}
		if rc < minRC {
			minRC = rc
		}
	}
	if d := maxRC - minRC; d > 1e-2 {
		t.Fatalf("expected risk contributions roughly equal, got spread %f across %v", d, w)
	}
}
