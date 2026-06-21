package main

import (
	"math"
	"testing"

	"github.com/akryanblue-eng/chicago-chase/pkg/quantumstar/optimizer"
)

// Baseline objective values captured for the fixed seed/config this demo
// runs with. They're deterministic by construction (fixed seed), so any
// drift here means the optimizer, an objective, or the schedule changed —
// intentionally or not. Update these alongside such a change.
const (
	baselineMinVarianceCost = 0.0078932539738556969
	baselineRiskParityCost  = 7.9144667041354877e-06
	driftTolerance          = 1e-9
)

func TestObjectiveBaselineDrift(t *testing.T) {
	start := []float64{0.25, 0.25, 0.25, 0.25}

	minVar := optimizer.Anneal(portfolioVariance, start, bounds(), config(1))
	riskParity := optimizer.Anneal(optimizer.RiskParityObjective(covariance, 10, 100, 10), start, bounds(), config(1))

	checkDrift(t, "min-variance objective", minVar.Cost, baselineMinVarianceCost)
	checkDrift(t, "risk-parity objective", riskParity.Cost, baselineRiskParityCost)
}

func checkDrift(t *testing.T, name string, current, baseline float64) {
	delta := current - baseline
	t.Logf("%s: current=%.17g baseline=%.17g delta=%.3e", name, current, baseline, delta)
	if math.Abs(delta) > driftTolerance {
		t.Errorf("%s drifted beyond tolerance: delta=%.3e (tolerance=%.3e)", name, delta, driftTolerance)
	}
}
