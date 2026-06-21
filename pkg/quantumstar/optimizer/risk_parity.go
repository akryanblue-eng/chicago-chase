package optimizer

const riskParityEpsilon = 1e-8
const riskParityWMax = 1.0

// RiskParityObjective builds an Objective that drives portfolio weights
// toward equal risk contribution across assets, subject to long-only,
// fully-invested constraints enforced as penalties. cov is the asset
// covariance matrix; lambdaSum, lambdaNeg, and lambdaUpper weight the
// sum-to-one, non-negativity, and upper-bound penalties respectively.
//
// The parity term is normalized by trace(cov) rather than portfolio
// variance: trace is invariant to w, which avoids a feedback loop where
// shrinking variance (the denominator) amplifies the parity term itself —
// a degenerate corner that variance-normalization is prone to when cov is
// ill-conditioned.
func RiskParityObjective(cov [][]float64, lambdaSum, lambdaNeg, lambdaUpper float64) ObjectiveFunc {
	n := len(cov)
	var trace float64
	for i := 0; i < n; i++ {
		trace += cov[i][i]
	}

	return func(w []float64) float64 {
		variance := quadForm(w, cov)
		mrc := matVec(cov, w)

		target := variance / float64(n)
		var parity float64
		for i, wi := range w {
			d := wi*mrc[i] - target
			parity += d * d
		}
		parity /= trace + riskParityEpsilon

		var sum, negPenalty, upperPenalty float64
		for _, wi := range w {
			sum += wi
			if wi < 0 {
				negPenalty += wi * wi
			}
			if wi > riskParityWMax {
				d := wi - riskParityWMax
				upperPenalty += d * d
			}
		}
		sumPenalty := (sum - 1) * (sum - 1)

		return parity + lambdaSum*sumPenalty + lambdaNeg*negPenalty + lambdaUpper*upperPenalty
	}
}
