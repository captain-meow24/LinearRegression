# CMake generated Testfile for 
# Source directory: /home/kanishka/Desktop/LinearRegression
# Build directory: /home/kanishka/Desktop/LinearRegression/cmake-build-debug
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[run_example]=] "/usr/bin/qemu-riscv64" "-L" "/home/kanishka/Downloads/riscv/sysroot" "/home/kanishka/Desktop/LinearRegression/cmake-build-debug/example")
set_tests_properties([=[run_example]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/kanishka/Desktop/LinearRegression/CMakeLists.txt;28;add_test;/home/kanishka/Desktop/LinearRegression/CMakeLists.txt;0;")
add_test([=[run_example2]=] "/usr/bin/qemu-riscv64" "-L" "/home/kanishka/Downloads/riscv/sysroot" "/home/kanishka/Desktop/LinearRegression/cmake-build-debug/example2")
set_tests_properties([=[run_example2]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/kanishka/Desktop/LinearRegression/CMakeLists.txt;32;add_test;/home/kanishka/Desktop/LinearRegression/CMakeLists.txt;0;")
add_test([=[run_example3]=] "/usr/bin/qemu-riscv64" "-L" "/home/kanishka/Downloads/riscv/sysroot" "/home/kanishka/Desktop/LinearRegression/cmake-build-debug/example3")
set_tests_properties([=[run_example3]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/kanishka/Desktop/LinearRegression/CMakeLists.txt;36;add_test;/home/kanishka/Desktop/LinearRegression/CMakeLists.txt;0;")
