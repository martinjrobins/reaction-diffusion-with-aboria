cmake_minimum_required(VERSION 2.8)

include(CMakeToolsHelpers OPTIONAL)
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

# Python
# Note: python version must match the python version that Boost Numpy
#       is compiled with, other than this choose whichever version you wish
find_package(PythonLibs 2.7 EXACT REQUIRED)
list(APPEND chemo_LIBRARIES ${PYTHON_LIBRARIES})
list(APPEND chemo_INCLUDES ${PYTHON_INCLUDE_DIRS})

# Boost
# Note: newer cmake versions can request numpy27 (https://gitlab.kitware.com/cmake/cmake/issues/16612)
#       to ensure that the python version matches the requested version above
find_package(Boost COMPONENTS numpy REQUIRED)
list(APPEND chemo_LIBRARIES ${Boost_LIBRARIES})
list(APPEND chemo_INCLUDES ${Boost_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
list(APPEND chemo_INCLUDES ${EIGEN3_INCLUDE_DIR})

# Aboria requires c++14
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++14")

# Aboria
set(Aboria_LOG_LEVEL 1 CACHE STRING "Logging level (1 = least, 3 = most)")
add_definitions(-DABORIA_LOG_LEVEL=${Aboria_LOG_LEVEL})
list(APPEND chemo_INCLUDES Aboria/src)
list(APPEND chemo_INCLUDES Aboria/third-party)

include_directories(${chemo_INCLUDES})

set(chemo_SOURCE
    python_wrapper.cpp
    chemo_example.cpp
    )

set(chemo_HEADERS
    chemo_example.hpp
    )

PYTHON_ADD_MODULE (chemo ${chemo_SOURCE})
target_link_libraries(chemo ${chemo_LIBRARIES})

