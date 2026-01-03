#!/bin/bash
g++ -Wall -fPIC -fopenmp -shared `python3 -m pybind11 --includes` -I ../../include -I/usr/include/eigen3 pyopenGJK.cpp ../../src/openGJK.c -o opengjkc`python3-config --extension-suffix`