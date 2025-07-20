cmake_minimum_required(VERSION 3.14)
project(anomaly_detector_core)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Add the module
pybind11_add_module(anomaly_detector_core 
    cpp_src/anomaly_detector_core.cpp
)

# Set properties
target_compile_features(anomaly_detector_core PRIVATE cxx_std_17)

# Compiler-specific options
if(MSVC)
    target_compile_options(anomaly_detector_core PRIVATE /W4 /O2)
else()
    target_compile_options(anomaly_detector_core PRIVATE -Wall -Wextra -O3 -march=native)
endif()

# Installation
install(TARGETS anomaly_detector_core DESTINATION ${Python_SITELIB})