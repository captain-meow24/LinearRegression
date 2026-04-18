cmake_minimum_required(VERSION 3.26)
project(LinearRegression)

set(CMAKE_CXX_STANDARD 17)

add_executable(LinearRegression example.cpp LinearRegressor.cpp)

target_compile_options(LinearRegression PRIVATE -O3 -mavx2)