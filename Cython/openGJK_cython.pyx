#cython: language_level=3, boundscheck=False

import numpy as np
from libc.stdlib cimport free, malloc
from cpython.mem  cimport PyMem_Malloc, PyMem_Free


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

# Create Python function
def pygjk(bod1, bod2):

	# Declare data types
	cdef: 
		simplex s
		bd bd1
		bd bd2
		double dist2

	# Convert 1D array to 2D, if any
	if bod1.ndim < 2:
		bod1 = np.append([bod1], [[1.,1.,1.]], axis = 0)
		bd1.numpoints = np.size(bod1,0) - 1
	else:
		bd1.numpoints = np.size(bod1,0)


	if bod2.ndim < 2:
		bod2 = np.append([bod2], [[1.,1.,1.]], axis = 0)
		bd2.numpoints = np.size(bod2,0) - 1
	else:
		bd2.numpoints = np.size(bod2,0)

	
	# Allocate memory for bodies
	bd1.coord = <double **> malloc(bd1.numpoints * sizeof(double *))
	if not bd1.coord:
		raise NameError('Not enough memory for bd1.coord')
	for i in range(0, bd1.numpoints):
		bd1.coord[i] = <double *> malloc(3 * sizeof(double))
		if not bd1.coord[i]:
			raise NameError('Not enough memory for bd1.coord[]')

	bd2.coord = <double **> malloc(bd2.numpoints * sizeof(double *))
	if not bd2.coord:
		raise NameError('Not enough memory for bd2.coord')
	for j in range(0, bd2.numpoints):
		bd2.coord[j] = <double *> malloc(3 * sizeof(double))
		if not bd2.coord[j]:
			raise NameError('Not enough memory for bd2.coord[]')

	# Create numpy-array MemoryView
	cdef:	
		double [:,:] narr1 = bod1 	
		double [:,:] narr2 = bod2

	# Assign coordinate values
	for i in range(0, bd1.numpoints):
		for j in range(0,3):
			bd1.coord[i][j] = narr1[i,j]
	
	for i in range(0, bd2.numpoints):
		for j in range(0,3):
			bd2.coord[i][j] = narr2[i,j]

	# Call C function
	dist2 = gjk(bd1, bd2, &s)

	# Free the memory
	for ii in range(0, bd1.numpoints):
		free(bd1.coord[ii])
	free(bd1.coord)

	for jj in range(0, bd2.numpoints):
		free(bd2.coord[jj])
	free(bd2.coord)

	return dist2

