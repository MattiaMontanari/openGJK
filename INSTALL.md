# How to compile openGJK

Using openGJK is very simple. This guide will help you getting started compiling and using openGJK.

## Requirements

### Common requirements

1. A C compiler
2. [CMake](http://www.cmake.org) version 3.5 or above

## Building
First, you need to configure the compilation, using CMake. 

1. Go inside the `build` dir. Create it if it doesn't exist.
2. Move into `build` dir and use `cmake ..`. On Windows you can specify  `cmake -G "Visual Studio 15 2017 Win64" ..`, on Unix `cmake -G "Unix Makefiles" ..`.
 
### CMake standard options

- CMAKE_BUILD_TYPE:     The type of build (can be Debug or Release)
- CMAKE_C_COMPILER:     The path to the C compiler

### CMake options defined for openGJK

Options are defined in the following files:

- CmakeOptions.cmake

They can be changed with the -D option:

`cmake -DVERSION_ACCURATE=ON ..`

In addition to passing options on the command line, you can browse and edit
CMake options using `cmakesetup` (Windows), `cmake-gui` or `ccmake` (GNU/Linux
and MacOS X).

- Go to the build dir
- On Windows: run `cmakesetup`
- On GNU/Linux and MacOS X: run `ccmake ..`

### Install and run

If all above building commands were executed from `build`, the openGJK library can be found in the `build/src` directory.
You can run the binaries in `build/examples/*`.

To install the library copy the header file openGJK.h and the binaries in a folder accessible in the search path by all users (on Unix this would normally be /usr/local).

## Testing

 TO REWRITE!!

As mention above you can turn on the unit tests and make it possible to easily
execute them:

`cmake -DCMAKE_BUILD_TYPE=Debug -DUNIT_TESTING=ON ..`

After that you can simply call `make test` in the build directory or if you
want more output simply call `ctest -V`.

If you want to enable the generation of coverage files you can do this by
using the following options:

`cmake -DCMAKE_BUILD_TYPE=Profiling -DUNIT_TESTING=ON ..`

After building it you will see that you have several coverage options in

`make help`

You should have `make ExperimentalCoverage` and running it will create
coverage files. The result is stored in Testing directory.

## Examples


This section presents three examples on how to use openGJK with C, C# and Matlab. 

### C
This example illustrates how to include openGJK in an existing C
 program.

All files for the example are in the `example1_c` folder. The executable built with
 CMake reads the coordinates of two polytopes from the input files,
 respectively userP.dat and userQ.dat, and computes the minimum distance
 between them.

Notice that the input files must be in the folder from which the executable
 is launched, otherwise an error is returned.
 
You can edit the coordinates in the input file to test different
 polytopes; just remember to edit also the first number in the files
 that corresponds to the numbers of vertices that the program will read.
 
### Matlab
This example illustrates how to invoke openGJK as a regular built-in
 Matlab function. You will need to build mex files (find out the requisites from [Mathworks documentation](https://uk.mathworks.com/help/matlab/matlab_external/what-you-need-to-build-mex-files.html)).


Open Matlab and cd into the `example2_mex` folder. By running the
 script `runme.m`, Matlab will first compile a mex file (telling you
 about the name of the mex file generated) and will call the script
 `main.m`. This invokes openGJK within Matlab and illustrates the
 result.

The mex file may be copied and called from any other Matlab project.

### C# #
This example illustrates how to invoke openGJK in an applications written in C#.  You will need [mono](http://www.mono-project.com/) and Microsoft Visual Studio toolchain for C# on Windows.

The only file required is in the `example3_csharp` folder. This can be compiled in Unix
 with mono, or in Windows using Visual Studio. Notice that, however, the openGJK library
 is compiled for a specific architecture (usually x64), and this breaks the portability 
 of the .NET application compiled in this example.

Below are the steps for compiling the C# application on Windows and Linux. Both 
 procedures assume the dynamic library of openGJK has been already compiled.

#### Compile on Windows
 1. Move into the folder `example3_csharp` and create a new folder `example3`.
 2. Copy into this folder the openGJK library or make it available in any directory.
 3. Open Visual Studio and create a new project. As project type select **Console App (.NET Framework)**.
 4. Add to this project the `main.cs` file
 5. Set x64 as the target platform, compile the application and run it.


#### Compile on Linux
 1. Move into the folder `example3_csharp` and create a new folder `example3`.
 2. Copy into this folder the openGJK library or install is so that is available in any directory.
 3. Move into that new folder and open a terminal.
 4. Type `mcs -out:example3demo  -d:UNIX ../main.cs`
 5. Run the example by typing `mono example3demo`

## API user reference

```double gjk( struct bodyA, struct bodyB, struct simplex)```

### Documentation
The folder `doc` contains a Doxygen file for generating the documentation of the whole
 library. To build the documentation cd into `doc` and call Doxygen from the command line simply by typing `doxygen`. If correctly installed, Doxygen will create html documentation with graphs illustrating the call stack of the functions of the library.

### Parameters
* **bodyA** The first body.
* **bodyB** The second body.
* **simplex** The simplex used the GJK algorithm at the first iteration.

### Returns
* **double** the minimum distance between bodyA and bodyB.

### Description
The function `gjk` computes the minimum Euclidean distance between two bodies using the 
 GJK algorithm. Note that the simplex used at the first iteration may be initialised by the user, but this is not necessary. 
 