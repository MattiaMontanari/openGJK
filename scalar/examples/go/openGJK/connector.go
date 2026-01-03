package openGJK

/*
#cgo CFLAGS: -I../../../include/ -I../../../
#cgo LDFLAGS: -lm
#include "openGJK.c"
void __array_to_matrix(double* input, unsigned long length, double* matrix[]) {
    for (int i = 0; i < length; i++) {
        double* row[3];
        for (int j = 0; j < 3; j++) {
            row[j] = &input[i*3+j];
        }
        matrix[i] = row[0];
    }
}
double __gjk(double *a, unsigned long a_length, double *b, unsigned long b_length) {
    gkSimplex simplex;
    gkPolytope a_polytope;
    gkPolytope b_polytope;
    double* a_matrix[a_length];
    double* b_matrix[b_length];
    __array_to_matrix(a, a_length, a_matrix);
    __array_to_matrix(b, b_length, b_matrix);
    a_polytope.coord = a_matrix;
    b_polytope.coord = b_matrix;
    a_polytope.numpoints = a_length;
    b_polytope.numpoints = b_length;
    double distance = compute_minimum_distance(a_polytope, b_polytope, &simplex);
    return distance;
}
*/
import "C"
import "unsafe"

// Converts nx3 matrix to n*3 array of C.double type
// Returns pointer to array first element
func matrix_to_carray(data [][3]float64) unsafe.Pointer {
	array := []C.double{}
	row_length := len(data)
	i := 0
	for i < row_length {
		j := 0
		for j < 3 {
			array = append(array, C.double(data[i][j]))
			j += 1
		}
		i += 1
	}
	return unsafe.Pointer(&array[0])
}

// Compute minimum distance of two objects
func GJK(a [][3]float64, b [][3]float64) float64 {
	pa := matrix_to_carray(a)
	pb := matrix_to_carray(b)
	na := C.ulong(len(a))
	nb := C.ulong(len(b))
	cdistance := C.__gjk((*C.double)(pa), na, (*C.double)(pb), nb)
	return float64(cdistance)
}
