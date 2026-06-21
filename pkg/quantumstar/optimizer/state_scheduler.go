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
func RunStateScheduled(first, second ObjectiveFunc, window int, threshold float64, chunkIterations, totalIterations int, start []float64, bounds Bounds, cfg Config) StateScheduleResult {
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
