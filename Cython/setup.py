
from distutils.core import setup, Extension
from Cython.Build import cythonize

exts = Extension(
		"openGJKpy", 
		sources = ["openGJK_cython.pyx", "openGJK.c"],
		# extra_compile_args = ['-fopenmp'],
		# extra_link_args = ['-fopenmp']
		)

setup(ext_modules = cythonize( [exts] ))


