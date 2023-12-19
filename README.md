# Filters for C++

Currently only implements the _Unscented Kalman Filter_: This is an almost line-by-line reimplementation of [`UnscentedKalmanFilter`](https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/UKF.py) from [FilterPy](https://github.com/rlabbe/filterpy) in C++17. For documentation, please refer to those references (for now, at least).

## Dependencies

The runtime dependencies are:
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page),
- [{fmt}](https://github.com/fmtlib/fmt) for formatting when throwing exceptions,

If you want to build and run the tests, you also need:
- [GoogleTest](https://github.com/google/googletest)
- [CMake](https://cmake.org/)

The library uses [Conan](https://conan.io/) for package management, which resolves all of the dependencies automatically.

## Setup

The build instructions below mainly concern package building and running the tests, since all library code is
contained in [`include/filter`](./include/filter).

### Installation

- *Manual Installation.* 

    You can just add [`include`](./include) to your project's include directories. Remember to also install the dependencies (i.e., [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) and [{fmt}](https://github.com/fmtlib/fmt)).
- *Install using Conan.*

    The commands below are meant to be executed from the libary's root directory.
    - Normal installation:
        ```
        $ conan create . -s compiler.cppstd=17 --build missing
        ```
    - Editable install:

        Install the dependencies using
        ```
        $ conan install .
        ```
        and then run
        ```
        $ conan editable add .
        ```
        To remove the editable install, use
        ```
        $ conan editable remove --refs=filter/0.1
        ```

### Usage

In your C++ file, add
```c
#include "filter/ukf.h"
#include "filter/sigma_points.h"
```
By default, this uses 64-bit double precision matrices/vectors. For 32-bit single precision matrices/vectors, use
```c
#define FILTER_USE_FP32
#include "filter/ukf.h"
#include "filter/sigma_points.h"
```
**Note:** This setting only affects the linear algebra types. Further, some tests will currently 
fail if they are run with single precision.

The library defines a general namespace `filter` and the namespace `filter::ukf` for the Unscented
Kalman Filter. See the [tests](./test/src/ukf.cpp) for usage examples.

### Building a Conan package
Run
```
$ conan build .
```
to build using the build type set in your Conan profile, or
```
$ conan build . --settings=build_type=BUILD_TYPE
```
with `BUILD_TYPE` substituted by your desired build type (e.g. `Debug` or `Release`).

## Tests
The tests run automatically after building. If the package is already installed, run
```
$ conan test test filter/0.1
```

## License
The library is licensed under the MIT license.

## References
[1] [FilterPy](https://github.com/rlabbe/filterpy)
