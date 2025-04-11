cmake_minimum_required(VERSION 3.10)
project(RustCppIntegration)

set(CMAKE_CXX_STANDARD 14)

# Define paths for Rust library
set(RUSTLIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rustlib)
set(RUSTLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rustlib/target/cxxbridge)
set(RUSTLIB_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rustlib/target/debug)

# Custom target to build the Rust library
add_custom_target(
    rust_lib
    COMMAND cargo build
    WORKING_DIRECTORY ${RUSTLIB_DIR}
    COMMENT "Building Rust library"
)

# C++ executable
add_executable(myapp main.cpp)

# Add include directories for the cxx bridge headers
target_include_directories(myapp PRIVATE ${RUSTLIB_INCLUDE_DIR})

# Link with the Rust library
target_link_directories(myapp PRIVATE ${RUSTLIB_LIB_DIR})
target_link_libraries(myapp PRIVATE
    rustlib
    pthread  # Might be needed depending on your platform
    dl       # Might be needed depending on your platform
)

# Make sure the Rust library is built before the C++ executable
add_dependencies(myapp rust_lib)
