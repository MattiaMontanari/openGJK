#cython: language_level=3, boundscheck=False
import numpy as np 

cdef extern from "openGJK.h":
	struct bd: 
		int numpoints
		double s[3]  		
		double** coord

	struct simplex:
		int nvrtx       	
		double vrtx[4][3]  	
		int wids[4]   
		double lambdas[4]

	double gjk(bd bd1, bd bd2, simplex *s)

def pygjk(bod1, bod2):
	"""Returns distance between two bodies, input: array of nodal coordinates for each body"""

	cdef: 
		simplex s
		bd bd1
		bd bd2
		double v1[3], v2[3]
		int i, j

	dummyRow = np.array([0.,0.,0.])

	bd1.numpoints = np.size(bod1, 0)
	if bd1.numpoints < 2:
		bod1 = np.vstack([bod1, dummyRow])
	for i in range(0, bd1.numpoints):
		v1[0] = bod1[i,0]
		v1[1] = bod1[i,1]
		v1[2] = bod1[i,2]
		bd1.coord[i] = v1
	
	bd2.numpoints = np.size(bod2, 0)
	if bd2.numpoints < 2:
		bod2 = np.vstack([bod2, dummyRow])
	for j in range(0, bd2.numpoints):
		v2[0] = bod2[j,0]
		v2[1] = bod2[j,1]
		v2[2] = bod2[j,2]
		bd2.coord[i] = v2

	answer = gjk(bd1, bd2, &s)



	return answer