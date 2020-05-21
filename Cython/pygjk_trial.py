import numpy as np
import openGJKpy as opengjk
a = np.array([[1.,1.,1.],[1.,1.,1.]])
b = np.array([[11.,1.,1.],[1.,1.,1.]])
d = opengjk.pygjk(a,b)
print(d)