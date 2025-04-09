#include <fmt/core.h>
#include <fmt/color.h>

int main() {
    fmt::print("Hello, {}!\n", "world");
    fmt::print(fg(fmt::color::crimson) | fmt::emphasis::bold,
               "This is a colorful {} message\n", "formatted");
    fmt::print("The answer is {}\n", 42);
    fmt::print("Pi is approximately {:.5f}\n", 3.14159265359);
    fmt::print("The answer is {}\n", 42);
    return 0;
}
