// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "EmailMemory",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "EmailMemoryCore",
            targets: ["EmailMemoryCore"]
        ),
        .executable(
            name: "EmailMemoryDemo",
            targets: ["EmailMemoryDemo"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "EmailMemoryCore"
        ),
        .executableTarget(
            name: "EmailMemoryDemo",
            dependencies: ["EmailMemoryCore"]
        )
    ]
)
