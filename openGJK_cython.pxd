# Declare C function and data types
cdef extern from "openGJK.h":
	struct bd:
		int numpoints
		double s[3]
		double ** coord

	struct simplex:
		int nvrtx
		double vrtx[4][3]
		int wids[4]
		double lambdas[4]

	double gjk(bd bd1, bd bd2, simplex *s)
