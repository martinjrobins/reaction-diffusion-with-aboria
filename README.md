About
-----

This is the supplementary code repository for the paper:

Maria Bruna, Philip K. Maini and Martin Robinson, "Particle-based simulations of
reaction-diffusion processes with Aboria: Application to a heterogeneous
population of cells with chemotaxis and volume exclusion"

Pre-requisites
--------------

Requires a C++14 compiler, Boost v1.65, Eigen v3, and CMake v2.8.

For example, these can be installed on Ubuntu 18.04 using apt:

``` {.bash}
$ sudo apt install build-essential libboost-all-dev cmake libeigen3-dev
```

Installation
------------

First clone this repository:

``` {.bash}
$ git clone --recurse-submodules https://github.com/martinjrobins/reaction-diffusion-with-aboria
```

Then create a build directory `build` under the main source directory:

``` {.bash}
$ cd reaction-diffusion-with-aboria
$ mkdir build
$ cd build
```

Then configure and compile the C++ module

``` {.bash}
$ cmake ..
$ make
```

Finally, change to the main source directory and run the `paper_plots.py` script
in order to (a) run the simulations described in the paper, and (b) generate the
plots in the paper

``` {.bash}
$ cd ..
$ python paper_plots.py
```
