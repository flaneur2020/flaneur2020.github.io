cmake_minimum_required(VERSION 3.14)
project(my_fmt_project VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

find_package(fmt CONFIG REQUIRED)

add_executable(hello src/main.cpp)
target_link_libraries(hello PRIVATE fmt::fmt)
