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

On Linux, Mac or Windows, if you have a basic C/C++ toolchain installed (git, compiler and cmake) clone this repo:

``` bash
git clone https://github.com/MattiaMontanari/openGJK.git
```

Then use these commands to build and run an example:

``` bash
cmake -E make_directory build
cmake -E chdir build cmake -DCMAKE_BUILD_TYPE=Release .. 
cmake --build build 
cmake -E chdir build/examples/c ./example_lib_opengjk_ce
```

If you get no errors, the successfull output is:

> `Distance between bodies 3.653650`

However, if you do get an error - any error - please file a bug! Support requests are welcome too.

# Use OpenGJK in your project

The best source to learn how to use OpenGJK are the examples. They are listed [here](https://www.mattiamontanari.com/opengjk/docs/examples/) for C, C#, Go, Matlab and Python. I aim to publish few more for Julia and Unity.

Take a look at the `examples` folder in this repo and have fun. File a request if you wish to see more!

# Contribute

You are very welcome to:

- Create pull requests of any kind
- Let me know if you are using this library and find it useful
- Open issues with request for support because they will help you and many others
- Cite this repository ([a sweet GitHub feature](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files#about-citation-files)) or my paper: Montanari, M. et at, *Improving the GJK Algorithm for Faster and More Reliable Distance Queries Between Convex Objects* (2017). ACM Trans. Graph.