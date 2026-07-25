// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "XagentDesktopRelay",
  platforms: [.macOS(.v14)],
  products: [
    .executable(
      name: "xagent-desktop-relay",
      targets: ["XagentDesktopRelay"]
    )
  ],
  targets: [
    .executableTarget(
      name: "XagentDesktopRelay",
      path: "Sources"
    )
  ]
)
