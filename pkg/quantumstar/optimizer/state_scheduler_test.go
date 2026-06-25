package optimizer

import "testing"

// firstStallStep replays stallSlope incrementally over a growing prefix of
// costs, mirroring exactly how RunStateScheduled detects a stall one chunk
// at a time. It returns the chunk index of the first stall, or -1 if costs
// never stalls.
func firstStallStep(costs []float64, window int, threshold float64) int {
	for i := 1; i <= len(costs); i++ {
		if slope, ok := stallSlope(costs[:i], window); ok && slope < threshold {
			return i - 1
		}
	}
	return -1
}

func TestStallSlope_NoPrematureSwitchOnSteadyProgress(t *testing.T) {
	// Cost drops by 1 every chunk, so every windowed slope is constant
	// and comfortably above the threshold.
	costs := []float64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	if step := firstStallStep(costs, 2, 0.5); step != -1 {
		t.Fatalf("expected no switch on steady progress, got switch at step %d", step)
	}
}

func TestStallSlope_TriggersExactlyWhenProgressPlateaus(t *testing.T) {
	// Cost drops steadily through index 5, then plateaus. With window=2
	// the windowed slope is 2 while descending, dropping to 0 once the
	// plateau fully enters the window — below the 0.5 threshold for the
	// first time at index 7.
	costs := []float64{10, 9, 8, 7, 6, 5, 5, 5, 5, 5, 5}
	const window = 2
	const threshold = 0.5
	const wantStep = 7

	if step := firstStallStep(costs, window, threshold); step != wantStep {
		t.Fatalf("expected switch at step %d, got %d", wantStep, step)
	}
}

func TestRunStateScheduled_DeterministicForFixedSeed(t *testing.T) {
	minVar := curriculumMinVariance
	riskParity := RiskParityObjective(curriculumCov, 10, 100, 10)
	start := []float64{0.25, 0.25, 0.25, 0.25}
	cfg := curriculumConfig(1)

	r1 := RunStateScheduled(minVar, riskParity, 3, 1e-4, 500, 10000, start, curriculumBounds(), cfg, TemperaturePolicy{})
	r2 := RunStateScheduled(minVar, riskParity, 3, 1e-4, 500, 10000, start, curriculumBounds(), cfg, TemperaturePolicy{})

	if r1.SwitchStep != r2.SwitchStep {
		t.Fatalf("expected identical switch step for the same seed, got %d vs %d", r1.SwitchStep, r2.SwitchStep)
	}
	if len(r1.CostHistory) != len(r2.CostHistory) {
		t.Fatalf("expected identical cost history length, got %d vs %d", len(r1.CostHistory), len(r2.CostHistory))
	}
	for i := range r1.CostHistory {
		if r1.CostHistory[i] != r2.CostHistory[i] {
			t.Fatalf("expected identical cost history, got %v vs %v", r1.CostHistory, r2.CostHistory)
		}
	}
	if r1.Cost != r2.Cost {
		t.Fatalf("expected identical final cost, got %f vs %f", r1.Cost, r2.Cost)
	}
	for i := range r1.Solution {
		if r1.Solution[i] != r2.Solution[i] {
			t.Fatalf("expected identical final solution, got %v vs %v", r1.Solution, r2.Solution)
		}
	}
}

func TestTemperaturePolicy_RaisesTempOnStall(t *testing.T) {
	policy := TemperaturePolicy{Gain: 1, Min: 0, Max: 10}
	got := policy.nextTemp(1.0, -1) // slope -1: cost got worse, not just flat
	if got <= 1.0 {
		t.Fatalf("expected temperature above base on stall, got %f", got)
	}
}

func TestTemperaturePolicy_LowersTempOnStrongProgress(t *testing.T) {
	policy := TemperaturePolicy{Gain: 1, Min: 0, Max: 10}
	got := policy.nextTemp(1.0, 5) // slope 5: strong progress
	if got >= 1.0 {
		t.Fatalf("expected temperature below base on strong progress, got %f", got)
	}
}

func TestTemperaturePolicy_ClampsToBounds(t *testing.T) {
	policy := TemperaturePolicy{Gain: 1, Min: 0.1, Max: 2.0}

	if got := policy.nextTemp(1.0, 100); got != policy.Min {
		t.Fatalf("expected clamp to Min=%f on extreme progress, got %f", policy.Min, got)
	}
	if got := policy.nextTemp(1.0, -100); got != policy.Max {
		t.Fatalf("expected clamp to Max=%f on extreme stall, got %f", policy.Max, got)
	}
}

func TestRunStateScheduled_TemperaturePolicyDeterministicForFixedSeed(t *testing.T) {
	minVar := curriculumMinVariance
	riskParity := RiskParityObjective(curriculumCov, 10, 100, 10)
	start := []float64{0.25, 0.25, 0.25, 0.25}
	cfg := curriculumConfig(1)
	policy := TemperaturePolicy{Gain: 5, Min: 0.05, Max: 2.0}

	r1 := RunStateScheduled(minVar, riskParity, 3, 1e-4, 500, 10000, start, curriculumBounds(), cfg, policy)
	r2 := RunStateScheduled(minVar, riskParity, 3, 1e-4, 500, 10000, start, curriculumBounds(), cfg, policy)

	if r1.SwitchStep != r2.SwitchStep {
		t.Fatalf("expected identical switch step under temperature policy, got %d vs %d", r1.SwitchStep, r2.SwitchStep)
	}
	if r1.Cost != r2.Cost {
		t.Fatalf("expected identical final cost under temperature policy, got %f vs %f", r1.Cost, r2.Cost)
	}
	for i := range r1.Solution {
		if r1.Solution[i] != r2.Solution[i] {
			t.Fatalf("expected identical final solution under temperature policy, got %v vs %v", r1.Solution, r2.Solution)
		}
	}
}
