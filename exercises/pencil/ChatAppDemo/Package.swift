// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ChatAppDemo",
    platforms: [
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "ChatAppDemo",
            targets: ["ChatAppDemo"]),
    ],
    targets: [
        .target(
            name: "ChatAppDemo",
            path: "ChatAppDemo"),
    ]
)
