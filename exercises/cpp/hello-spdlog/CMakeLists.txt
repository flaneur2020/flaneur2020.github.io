cmake_minimum_required(VERSION 3.15)
project(spdlog_hello VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include vcpkg toolchain file if not already set
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

# Find spdlog package
find_package(spdlog REQUIRED)

# Add executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Link against spdlog
target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog) 
