## # **2020—Apple Silicon**

### Heterogeneous Computing

 Heterogeneous Computing 是 M1 成功的秘密，一个 M1 芯片有下面这些组成：

- Image processing circuitry
- Mathematical signal processors
- AI-accelerating neural engines
- Dedicated video encoder and decoders
- A secure enclave for encrypted storage
- 8 GPU cores with 128 parallel execution units
- 4 high-performance _[Firestorm](https://www.anandtech.com/show/16252/mac-mini-apple-m1-tested)_ [CPU cores](https://www.anandtech.com/show/16252/mac-mini-apple-m1-tested)
- 4 efficient, low-energy _Icestorm_ CPU cores

### Unified Memory Architecture

GPU 与 CPU 之间有统一的内存访问。要做到这一点，需要解决：

- CPU 和 GPU 之间内存的访问形式不同，CPU 倾向于访问小的少量字节，而 GPU 倾向于访问大量的 blob 数据
- GPU 发热量高

apple 给 GPU 和 CPU 分配了同样的内存和 L3 cache 共享；