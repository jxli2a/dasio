cmake_minimum_required(VERSION 3.15)
project(dasio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(_bandpass MODULE dasio/cpp/bandpass2d.cpp)
target_link_libraries(_bandpass PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS _bandpass DESTINATION dasio/cpp)
