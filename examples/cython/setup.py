from setuptools import Extension, setup
from Cython.Build import cythonize

exts = Extension(
	"openGJK_cython",
	sources = ["openGJK_cython.pyx"],
	extra_compile_args=['-Iinclude/','-fopenmp'],
	extra_link_args=['-fopenmp'],
)

setup(
	name='openGJK-cython-version',
	ext_modules = cythonize( [exts] )
)