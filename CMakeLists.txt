cmake_minimum_required(VERSION 3.26)
project(LinearRegression)

set(CMAKE_CXX_STANDARD 17)

# Use RISC-V compiler
set(CMAKE_CXX_COMPILER /home/kanishka/Downloads/riscv/bin/riscv64-unknown-linux-gnu-g++)

add_executable(LinearRegression example.cpp LinearRegressor.cpp)

# RISC-V vector flags
target_compile_options(LinearRegression PRIVATE
        -O3
        -march=rv64gcv
        -mabi=lp64d
)