package compute

// broadcastShape computes the NumPy-style broadcast output shape for two shapes.
// Shapes are aligned from the right; each dimension is max(a, b) where one must be 1.
func broadcastShape(a, b []int) []int {
	na, nb := len(a), len(b)
	ndim := na
	if nb > ndim {
		ndim = nb
	}
	out := make([]int, ndim)
	for i := range out {
		da, db := 1, 1
		if ai := na - ndim + i; ai >= 0 {
			da = a[ai]
		}
		if bi := nb - ndim + i; bi >= 0 {
			db = b[bi]
		}
		if da >= db {
			out[i] = da
		} else {
			out[i] = db
		}
	}
	return out
}
