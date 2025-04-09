# CMake Spdlog Hello World

This is a simple project demonstrating the use of spdlog with CMake and vcpkg.

## Prerequisites

- CMake (3.15 or higher)
- vcpkg
- A C++17 compatible compiler

## Building the Project

1. Make sure you have vcpkg installed and the VCPKG_ROOT environment variable set.

2. Create a build directory and navigate to it:
   ```bash
   mkdir build
   cd build
   ```

3. Configure the project with CMake:
   ```bash
   cmake ..
   ```

4. Build the project:
   ```bash
   cmake --build .
   ```

5. Run the executable:
   ```bash
   ./spdlog_hello
   ```

## Notes

- If vcpkg is not installed, you can get it from: https://github.com/Microsoft/vcpkg
- The project uses spdlog for logging, which will be automatically installed by vcpkg 