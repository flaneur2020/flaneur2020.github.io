#!/bin/bash

# Check if VCPKG_ROOT is set
if [ -z "$VCPKG_ROOT" ]; then
    echo "Error: VCPKG_ROOT environment variable is not set"
    echo "Please set it to your vcpkg installation directory"
    exit 1
fi

# Create and enter build directory
mkdir -p build
cd build

# Configure with CMake using vcpkg toolchain
cmake -B . -S .. \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

# Build the project
cmake --build .

# Return to original directory
cd ..

echo "Build complete! You can run the program with: ./build/spdlog_hello" 