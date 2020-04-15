
openGJK                         {#mainpage}
=======

OpenGJK implements a new version of the Gilbert-Johnson-Keerthi (GJK) algorithm to
 compute the minimum distance between convex polytopes. OpenGJK is a C library which was tested on Unix and Windows using different compilers for multi-threaded applications.

Detailed information about the algorithm see "[Improving the GJK Algorithm for Faster and More Reliable Distance
 Queries Between Convex Objects. ACM Trans. on Graph. 2017](https://dl.acm.org/citation.cfm?id=3083724)".


When should I use openGJK?
--------------------------

OpenGJK is designed with speed, accuracy and robustness in mind and is therefore suitable for engineering, robotics and computer graphics simulations.
Basically, openGJK can be used in any application where the distance between **any convex polytope** is required.

Compile and run
---------------

To compile the OpenGJK library create a build dir,
and in the build dir call 'cmake ..' followed by 'make'. More details can be found in the INSTALL file.

There are examples for C, C# and Matlab in the `examples` folder. The INSTALL file provides information on how to run the examples.

Repository content
------------------

This repository contains the following files and folders:

```
│   CMakeLists.txt
│   README.md
│   
├───doc
│       openGJKcustomfooter.html
│       openGJKcustomheader.html
│       openGJKcustomstyle.css
│       Doxyfile
│       oxfordLogo.jpg
│       
├───example1_c
│       CMakeLists.txt
│       main.c
│       userP.dat
│       userQ.dat
│       
├───example2_mex
│       main.m
│       runme.m
│       
├───example3_csharp
│       main.cs
│       
└───lib
    │   CMakeLists.txt
    │   
    ├───ext
    │       predicates.c
    │       predicates.h
    │       
    ├───include
    │   └───openGJK
    │           openGJK.h
    │           
    └───src
            openGJK.c
```

More information
----------------

[OpenGJK](http://iel.eng.ox.ac.uk/?page_id=504) was developed at the Impact Engineering Laboratory, University of Oxford.


A clear presentation of the GJK algorithm can be found in the
 book by **Van der Bergen** *Collision Detection in Interactive 3D
 Environments*, edited by Elsevier.

More details about the GJK algorithm can be found in the original paper
 from Gilbert, Johnson and Keerthi [A fast procedure for computing the distance between complex objects in three-dimensional space](http://ieeexplore.ieee.org/document/2083/?arnumber=2083).


How to cite openGJK
-------------------

If you use openGJK for your research please cite [OpenGJK for C, C# and Matlab: Reliable solutions to distance queries between convex bodies in three-dimensional space. SoftwareX.  2018](https://www.sciencedirect.com/science/article/pii/S2352711018300591).

License
-------

This project is licensed undert the GNU General Public License v3.0.  
openGJK: open-source Gilbert-Johnson-Keerthi algorithm  
           Copyright (C) Mattia Montanari 2018 - 2019   
               http://iel.eng.ox.ac.uk/?page_id=504     