cd rustlib
cargo build

cd ..
mkdir build
cmake -B build -S .
cmake --build build
