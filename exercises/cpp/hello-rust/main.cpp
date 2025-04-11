#include "rustlib/src/lib.rs.h"
#include <iostream>

int main() {
    // Call Rust functions
    int sum = add(5, 7);
    std::cout << "5 + 7 = " << sum << std::endl;
    std::string message = greet("C++");
    std::cout << message << std::endl;
    return 0;
}

