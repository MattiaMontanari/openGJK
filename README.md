<!--                        _____      _ _  __                                      >
<                          / ____|    | | |/ /                                      >
<    ___  _ __   ___ _ __ | |  __     | | ' /                                       >
<   / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <                                        >
<  | (_) | |_) |  __/ | | | |__| | |__| | . \                                       >
<   \___/| .__/ \___|_| |_|\_____|\____/|_|\_\                                      >
<        | |                                                                        >
<        |_|                                                                        >
<                                                                                   >
< Copyright 2022 Mattia Montanari, University of Oxford                             >
<                                                                                   >
< This program is free software: you can redistribute it and/or modify it under     >
< the terms of the GNU General Public License as published by the Free Software     >
< Foundation, either version 3 of the License. You should have received a copy      >
< of the GNU General Public License along with this program. If not, visit          >
<                                                                                   >
<     https://www.gnu.org/licenses/                                                 >
<                                                                                   >
< This program is distributed in the hope that it will be useful, but WITHOUT       >
< ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS     >
< FOR A PARTICULAR PURPOSE. See GNU General Public License for details.           -->


# Get started

If you have some basic tools installed (git, compiler and cmake) clone this repo:

```
git clone https://github.com/MattiaMontanari/openGJK.git
```

followed by these commands:

```
cmake -E make_directory build
cmake -E chdir build cmake -DRUN_UNITESTS=ON  -DCMAKE_BUILD_TYPE=Release .. 
cmake --build build 
cmake -E chdir build/examples/c ./example_lib_opengjk_ce
cmake -E chdir "build/test" ctest --build-config Release
```

If you get no errors, the successfull output is:

> `Distance between bodies 3.653650`. 

However, if you do get an error - any error - please file a bug! Support requests are welcome too.

# Beyond getting started

With the commands above you have built a demo example tha invokes the openGJK library. The library is statically linked and the distance between two bodies is computed and returned. 

To learn how to use this library in your project the best place to start is the demo. Look at `main.c` and the other examples. In `examples/c/CMakeLists.txt` you can find how simple is to link using CMake.