package main

import (
	"fmt"

	"github.com/MattiaMontanari/openGJK/examples/go/openGJK"
)

func main() {
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
	collided := openGJK.GJK(a, b)
	if collided == 0 {
		fmt.Println("a and b is collided")
	}
	distance := openGJK.GJK(a, c)
	if distance > 0 {
		fmt.Println("distance from a to c is", distance)
	}
}
