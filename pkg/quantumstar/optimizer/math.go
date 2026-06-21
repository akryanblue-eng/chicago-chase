package optimizer

func dot(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func matVec(matrix [][]float64, v []float64) []float64 {
	result := make([]float64, len(v))
	for i := range matrix {
		result[i] = dot(matrix[i], v)
	}
	return result
}

func quadForm(v []float64, matrix [][]float64) float64 {
	return dot(v, matVec(matrix, v))
}
