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
        .executableTarget(
            name: "gemm-tiled",
            resources: [
                .process("Resources"),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
            ]
        ),
    ]
)
