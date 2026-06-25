package optimizer

// StateScheduleResult is the outcome of a state-dependent scheduling run:
// the underlying Anneal Result, the best cost recorded at the end of each
// chunk, and the chunk index at which the objective switched (-1 if it
// never did).
type StateScheduleResult struct {
	Result
	CostHistory []float64
	SwitchStep  int
}

// TemperaturePolicy biases the InitialTemp of the next chunk using the
// most recent stallSlope, clamped to [Min, Max]. It's evaluated once per
// chunk boundary, between Anneal calls — never mid-run — so it never
// reaches into the kernel. The zero value disables biasing entirely.
type TemperaturePolicy struct {
	Gain float64
	Min  float64
	Max  float64
}

// nextTemp biases baseTemp inversely to slope: stallSlope is the cost
// improvement over the trailing window, large when progress is strong
// and near zero (or negative) when stalled, so subtracting Gain*slope
// raises temperature on stall and lowers it on strong progress.
func (p TemperaturePolicy) nextTemp(baseTemp, slope float64) float64 {
	t := baseTemp - p.Gain*slope
	if t < p.Min {
		return p.Min
	}
	if t > p.Max {
		return p.Max
	}
	return t
}

// RunStateScheduled runs first under chunked annealing, recording the best
// cost at the end of each chunk, and switches permanently to second the
// first time the trailing window's cost improvement drops below
// threshold. Chunking is the only way to observe a cost trajectory
// without touching Anneal: each chunk is an ordinary Anneal call of
// chunkIterations, carrying the prior chunk's best solution forward as
// the next chunk's start. The switch is one-way — once triggered, every
// remaining chunk runs under second. Each chunk after the first offsets
// cfg.Seed by its chunk index, mirroring RunCurriculum's phase-seed
// convention, so it doesn't replay an earlier chunk's random sequence.
//
// If tempPolicy is non-zero, each chunk after the window has filled
// seeds its InitialTemp from the prior chunk's stallSlope via
// tempPolicy.nextTemp, instead of reusing cfg.InitialTemp unchanged.
// This only ever sets a chunk's starting temperature before Anneal is
// called — Anneal itself stays unaware any of this exists.
func RunStateScheduled(first, second ObjectiveFunc, window int, threshold float64, chunkIterations, totalIterations int, start []float64, bounds Bounds, cfg Config, tempPolicy TemperaturePolicy) StateScheduleResult {
	objective := first
	switchStep := -1
	current := start

	var costs []float64
	var last Result

	remaining := totalIterations
	for chunk := 0; remaining > 0; chunk++ {
		iters := chunkIterations
		if iters > remaining {
			iters = remaining
		}

		chunkCfg := cfg
		chunkCfg.MaxIterations = iters
		chunkCfg.Seed = cfg.Seed + int64(chunk)

		if slope, ok := stallSlope(costs, window); ok && tempPolicy != (TemperaturePolicy{}) {
			chunkCfg.InitialTemp = tempPolicy.nextTemp(cfg.InitialTemp, slope)
		}

		last = Anneal(objective, current, bounds, chunkCfg)
		current = last.Solution
		costs = append(costs, last.Cost)

		if switchStep == -1 {
			if slope, ok := stallSlope(costs, window); ok && slope < threshold {
				objective = second
				switchStep = chunk
			}
		}

		remaining -= iters
	}

	return StateScheduleResult{Result: last, CostHistory: costs, SwitchStep: switchStep}
}

// stallSlope returns the cost improvement over the trailing window —
// costs[len-1-window] minus costs[len-1] — and whether enough history
// exists yet to compute it. It depends only on the cost history, so a
// given (costs, window) pair always yields the same slope: this is the
// scheduler's only state, and it's a pure function of it.
func stallSlope(costs []float64, window int) (float64, bool) {
	n := len(costs)
	if n <= window {
		return 0, false
	}
	return costs[n-1-window] - costs[n-1], true
}
