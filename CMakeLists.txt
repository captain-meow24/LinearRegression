cmake_minimum_required(VERSION 3.31)
project(LinearRegression)

set(CMAKE_CXX_STANDARD 20)

add_executable(LinearRegression
        example.cpp
        LinearRegressor.h
        LinearRegressor.cpp)
