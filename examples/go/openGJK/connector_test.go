package openGJK

import "testing"

func fassert(t *testing.T, got, want float64) {
	if got != want {
		t.Errorf("GJK(b, c) -> %f, want %f", got, want)
	}
}

func TestGJK(t *testing.T) {
	a := [][3]float64{
		{-1.0, -1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{1.0, 1.0, 0.0},
		{1.0, -1.0, 0.0},
	}
	b := [][3]float64{
		{0.0, -0.5, 0.0},
		{0.0, 0.5, 0.0},
		{2.0, 0.5, 0.0},
		{2.0, -0.5, 0.0},
	}
	c := [][3]float64{
		{3.0, -0.5, 0.0},
		{3.0, 0.5, 0.0},
		{5.0, 0.5, 0.0},
		{5.0, -0.5, 0.0},
	}
	fassert(t, GJK(a, b), 0)
	fassert(t, GJK(b, c), 1)
	fassert(t, GJK(a, c), 2)
}
