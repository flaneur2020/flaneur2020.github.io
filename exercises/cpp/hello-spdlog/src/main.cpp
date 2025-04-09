#include <spdlog/spdlog.h>

int main() {
    spdlog::info("Hello, World!");
    
    // Example of different log levels
    spdlog::debug("This is a debug message");
    spdlog::warn("This is a warning message");
    spdlog::error("This is an error message");
    
    return 0;
} 