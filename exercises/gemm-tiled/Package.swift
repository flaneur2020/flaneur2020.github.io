// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "gemm-tiled",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(
            name: "gemm-tiled",
            targets: ["gemm-tiled"]
        ),
    ],
    targets: [
        .target(
            name: "Metal4Interop",
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedFramework("Metal"),
            ]
        ),
        .executableTarget(
            name: "gemm-tiled",
            dependencies: ["Metal4Interop"],
            resources: [
                .process("Resources"),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
            ]
        ),
    ]
)
