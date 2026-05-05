cmake_minimum_required(VERSION 3.26)
project(LinearRegression)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_COMPILER /home/kanishka/Downloads/riscv/bin/riscv64-unknown-linux-gnu-g++)

enable_testing()

set(RVV_FLAGS
        -O3
        -march=rv64gcv
        -mabi=lp64d
)

add_executable(example example.cpp LinearRegressor.cpp)
target_compile_options(example PRIVATE ${RVV_FLAGS})

add_executable(example2 example2.cpp LinearRegressor.cpp)
target_compile_options(example2 PRIVATE ${RVV_FLAGS})

add_executable(example3 example3.cpp LinearRegressor.cpp)
target_compile_options(example3 PRIVATE ${RVV_FLAGS})

set(QEMU /usr/bin/qemu-riscv64)
set(SYSROOT /home/kanishka/Downloads/riscv/sysroot)

add_test(NAME run_example
        COMMAND ${QEMU} -L ${SYSROOT} $<TARGET_FILE:example>
)

add_test(NAME run_example2
        COMMAND ${QEMU} -L ${SYSROOT} $<TARGET_FILE:example2>
)

add_test(NAME run_example3
        COMMAND ${QEMU} -L ${SYSROOT} $<TARGET_FILE:example3>
)