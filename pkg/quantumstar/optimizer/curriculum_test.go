package optimizer

import "testing"

var curriculumCov = [][]float64{
	{0.040, 0.010, 0.004, 0.002},
	{0.010, 0.030, 0.006, 0.003},
	{0.004, 0.006, 0.020, 0.001},
	{0.002, 0.003, 0.001, 0.015},
}

func curriculumBounds() Bounds {
	return Bounds{Min: []float64{0, 0, 0, 0}, Max: []float64{1, 1, 1, 1}}
}

func curriculumConfig(seed int64) Config {
	return Config{InitialTemp: 1.0, CoolingRate: 0.999, MaxIterations: 0, Seed: seed}
}

// curriculumMinVariance mirrors the demo's long-only, fully-invested
// min-variance objective; quadForm alone has no penalty against the
// trivial w=0 minimum, so it can't stand in for a real portfolio objective.
func curriculumMinVariance(w []float64) float64 {
	variance := quadForm(w, curriculumCov)

	sum := 0.0
	for _, wi := range w {
		sum += wi
		if wi < 0 {
			variance += 10 * wi * wi
		}
	}
	variance += 10 * (sum - 1) * (sum - 1)

	return variance
}

func TestRunCurriculum_FullFirstPhaseMatchesDirectAnneal(t *testing.T) {
	minVar := curriculumMinVariance
	riskParity := RiskParityObjective(curriculumCov, 10, 100, 10)
	scheduler := LinearCurriculum(minVar, riskParity, 1.0)
	start := []float64{0.25, 0.25, 0.25, 0.25}

	cfg := curriculumConfig(1)
	cfg.MaxIterations = 5000

	got := RunCurriculum(scheduler, 1.0, start, curriculumBounds(), cfg, 5000)
	want := Anneal(minVar, start, curriculumBounds(), cfg)

	if got.Cost != want.Cost {
		t.Fatalf("expected switchAt=1.0 to match a direct min-var Anneal, got cost %f want %f", got.Cost, want.Cost)
	}
	for i := range got.Solution {
		if got.Solution[i] != want.Solution[i] {
			t.Fatalf("expected switchAt=1.0 to match a direct min-var Anneal, got %v want %v", got.Solution, want.Solution)
		}
	}
}

func TestRunCurriculum_SecondPhaseSatisfiesRiskParityInvariants(t *testing.T) {
	minVar := curriculumMinVariance
	riskParity := RiskParityObjective(curriculumCov, 10, 100, 10)
	scheduler := LinearCurriculum(minVar, riskParity, 0.0)
	start := []float64{0.25, 0.25, 0.25, 0.25}

	cfg := curriculumConfig(1)
	cfg.MaxIterations = 20000
	result := RunCurriculum(scheduler, 0.0, start, curriculumBounds(), cfg, 20000)

	assertLongOnlyFullyInvested(t, result.Solution)
	assertRiskContributionSpread(t, result.Solution, 1e-2)
}

func TestRunCurriculum_MidSwitchSatisfiesConstraints(t *testing.T) {
	minVar := curriculumMinVariance
	riskParity := RiskParityObjective(curriculumCov, 10, 100, 10)
	scheduler := LinearCurriculum(minVar, riskParity, 0.3)
	start := []float64{0.25, 0.25, 0.25, 0.25}

	cfg := curriculumConfig(1)
	cfg.MaxIterations = 20000
	result := RunCurriculum(scheduler, 0.3, start, curriculumBounds(), cfg, 20000)

	assertLongOnlyFullyInvested(t, result.Solution)
	assertRiskContributionSpread(t, result.Solution, 5e-2)
}

func TestRunCurriculum_DeterministicForFixedSeed(t *testing.T) {
	minVar := curriculumMinVariance
	riskParity := RiskParityObjective(curriculumCov, 10, 100, 10)
	scheduler := LinearCurriculum(minVar, riskParity, 0.3)
	start := []float64{0.25, 0.25, 0.25, 0.25}

	cfg := curriculumConfig(1)
	cfg.MaxIterations = 20000

	r1 := RunCurriculum(scheduler, 0.3, start, curriculumBounds(), cfg, 20000)
	r2 := RunCurriculum(scheduler, 0.3, start, curriculumBounds(), cfg, 20000)

	if r1.Cost != r2.Cost {
		t.Fatalf("expected identical cost for the same seed, got %f vs %f", r1.Cost, r2.Cost)
	}
	for i := range r1.Solution {
		if r1.Solution[i] != r2.Solution[i] {
			t.Fatalf("expected identical solution for the same seed, got %v vs %v", r1.Solution, r2.Solution)
		}
	}
}

func assertLongOnlyFullyInvested(t *testing.T, w []float64) {
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
}

func assertRiskContributionSpread(t *testing.T, w []float64, tolerance float64) {
	mrc := matVec(curriculumCov, w)
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
	if d := maxRC - minRC; d > tolerance {
		t.Fatalf("expected risk contributions within %g, got spread %f across %v", tolerance, d, w)
	}
}
