#cython: language_level=3, boundscheck=False

import numpy as np
from libc.stdlib cimport free, malloc
from cpython.mem  cimport PyMem_Malloc, PyMem_Free


# Declare C function and data
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
		int i, j
		double answer

	print("Break 1")#--------------------------------------------------

	# Convert 1D array to 2D, if any
	if bod1.ndim < 2:
		bod1 = np.append([bod1], [[1.,1.,1.]], axis = 0)
		bd1.numpoints = np.size(bod1,0) - 1
	else:
		bd1.numpoints = np.size(bod1,0)

	print(bd1.numpoints)

	if bod2.ndim < 2:
		bod2 = np.append([bod2], [[1.,1.,1.]], axis = 0)
		bd2.numpoints = np.size(bod2,0) - 1
	else:
		bd2.numpoints = np.size(bod2,0)

	print(bd2.numpoints)

	
	# Allocate memory for pointer (not working)
	bd1.coord = <double **> malloc(bd1.numpoints * sizeof(double *))
	bd2.coord = <double **> malloc(bd2.numpoints * sizeof(double *))
		

	print("Break 2")#--------------------------------------------------
	
	# Create numpy-array MemoryView
	cdef:	
		double [:,:] narr1 = bod1 	
		double [:,:] narr2 = bod2

	print(narr2[0,0]) # output a <double>, works fine 

	print("Break 3")#--------------------------------------------------

	# Assign coordinate values (Segmentation Fault Here!!, )
	for i in range(0, bd1.numpoints):
		bd1.coord[i] = <double *> malloc(3 * sizeof(double))
		bd1.coord[i][0] = narr1[i,0]
		bd1.coord[i][1] = narr1[i,1]
		bd1.coord[i][2] = narr1[i,2]

	
	for j in range(0, bd2.numpoints):
		bd2.coord[j] = <double *> malloc(3 * sizeof(double))
		bd2.coord[j][0] = narr2[j,0]
		bd2.coord[j][1] = narr2[j,1]
		bd2.coord[j][2] = narr2[j,2]


	print("Break 4")#--------------------------------------------------

	# Call C function
	answer = gjk(bd1, bd2, &s)

	# Free the memory
	for i in range(0, bd1.numpoints):
		free(bd1.coord[i])
	for j in range(0, bd2.numpoints):
		free(bd2.coord[j])


	return answer

