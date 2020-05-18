
from distutils.core import setup, Extension
from Cython.Build import cythonize

exts = Extension(
		"openGJKpy", 
		sources = ["openGJK_cython.pyx", "openGJK.c"]
		)

setup(ext_modules = cythonize( [exts] ))


