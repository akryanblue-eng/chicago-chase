package optimizer

const riskParityEpsilon = 1e-8
const riskParityWMax = 1.0

// RiskParityObjective builds an Objective that drives portfolio weights
// toward equal risk contribution across assets, subject to long-only,
// fully-invested constraints enforced as penalties. cov is the asset
// covariance matrix; lambdaSum, lambdaNeg, and lambdaUpper weight the
// sum-to-one, non-negativity, and upper-bound penalties respectively.
//
// The parity term is normalized by portfolio variance so it stays
// shape-driven rather than scaling with the magnitude of cov.
func RiskParityObjective(cov [][]float64, lambdaSum, lambdaNeg, lambdaUpper float64) ObjectiveFunc {
	n := len(cov)

	return func(w []float64) float64 {
		variance := quadForm(w, cov)
		mrc := matVec(cov, w)

		target := variance / float64(n)
		var parity float64
		for i, wi := range w {
			d := wi*mrc[i] - target
			parity += d * d
		}
		parity /= variance + riskParityEpsilon

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
