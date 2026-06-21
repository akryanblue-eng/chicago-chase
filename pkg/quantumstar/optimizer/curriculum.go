package optimizer

// TimeStep is a deterministic, CI-safe point in an annealing run,
// expressed as an iteration index out of a total. There is no wall clock
// and no randomness here, so a schedule built on TimeStep stays
// reproducible under a fixed seed.
type TimeStep struct {
	Step  int
	Total int
}

// ObjectiveScheduler selects which Objective is active at a given
// TimeStep. It sits above objective selection — Anneal itself is never
// aware a schedule exists.
type ObjectiveScheduler func(t TimeStep) ObjectiveFunc

// LinearCurriculum returns a scheduler that selects first for the initial
// switchAt fraction of the run (switchAt in [0,1]) and second afterward.
func LinearCurriculum(first, second ObjectiveFunc, switchAt float64) ObjectiveScheduler {
	return func(t TimeStep) ObjectiveFunc {
		if float64(t.Step)/float64(t.Total) < switchAt {
			return first
		}
		return second
	}
}

// RunCurriculum runs a two-phase anneal under scheduler: phase one for
// switchAt*totalIterations steps, phase two for the remainder, carrying
// phase one's best solution forward as phase two's starting point. The
// annealer itself is untouched — each phase is an ordinary Anneal call
// under cfg, scoped to that phase's iteration budget. Phase two uses
// cfg.Seed+1 so it doesn't replay phase one's exact random sequence.
func RunCurriculum(scheduler ObjectiveScheduler, switchAt float64, start []float64, bounds Bounds, cfg Config, totalIterations int) Result {
	switchStep := int(switchAt * float64(totalIterations))

	phase1Cfg := cfg
	phase1Cfg.MaxIterations = switchStep
	phase1Objective := scheduler(TimeStep{Step: 0, Total: totalIterations})
	phase1 := Anneal(phase1Objective, start, bounds, phase1Cfg)

	phase2Cfg := cfg
	phase2Cfg.MaxIterations = totalIterations - switchStep
	phase2Cfg.Seed = cfg.Seed + 1
	phase2Objective := scheduler(TimeStep{Step: totalIterations - 1, Total: totalIterations})
	phase2 := Anneal(phase2Objective, phase1.Solution, bounds, phase2Cfg)

	return phase2
}
