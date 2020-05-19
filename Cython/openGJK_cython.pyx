#cython: language_level=3, boundscheck=False
import numpy as np 
from libc.stdlib cimport free

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

	print("Hello!!")
	cdef: 
		simplex ss
		bd bd1
		bd bd2
		int i, j
		double answer

	if bod1.ndim < 2:
		bod1 = np.append([bod1], [[1.,1.,1.]], axis = 0)
	if bod2.ndim < 2:
		bod2 = np.append([bod2], [[1.,1.,1.]], axis = 0)

	bd1.numpoints = np.size(bod1,0)
	bd2.numpoints = np.size(bod2,0)

	cdef:	
		double [:,:] narr1 = bod1 	# create numpy-array MemoryView
		double [:,:] narr2 = bod2

	
	for i in range(0, bd1.numpoints):
		bd1.coord[i] = hex(id(bod1[i]))

	
	for j in range(0, bd2.numpoints):
		bd2.coord[i][0] = narr2[j,0]
		bd2.coord[i][1] = narr2[j,1]
		bd2.coord[i][2] = narr2[j,2]


	answer = gjk(bd1, bd2, &ss)

	free(bd1.coord)
	free(bd2.coord)


	return answer

