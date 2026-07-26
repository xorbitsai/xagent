import AppKit
import CoreGraphics
import Foundation

struct CommandLineOptions {
  let setup: PairingSetup
  let windowID: CGWindowID?
  let displayID: CGDirectDisplayID?
  let configurationStore: DesktopRelayConfigurationStore
  let setupFileURL: URL?
  let shouldPersistConfiguration: Bool

  static func parse(
    _ arguments: [String],
    configurationStore: DesktopRelayConfigurationStore = .defaultStore()
  ) throws -> Self {
    var setupJSON: String?
    var setupFile: String?
    var windowID: CGWindowID?
    var displayID: CGDirectDisplayID?
    var index = 1
    while index < arguments.count {
      switch arguments[index] {
      case "--setup":
        index += 1
        guard index < arguments.count else {
          throw RelayFailure.invalidSetup("--setup requires JSON")
        }
        setupJSON = arguments[index]
      case "--setup-file":
        index += 1
        guard index < arguments.count else {
          throw RelayFailure.invalidSetup("--setup-file requires a path")
        }
        setupFile = arguments[index]
      case "--window-id":
        index += 1
        guard
          index < arguments.count,
          let parsed = CGWindowID(arguments[index])
        else {
          throw RelayFailure.invalidSetup(
            "--window-id requires an unsigned integer"
          )
        }
        windowID = parsed
      case "--display-id":
        index += 1
        guard
          index < arguments.count,
          let parsed = CGDirectDisplayID(arguments[index])
        else {
          throw RelayFailure.invalidSetup(
            "--display-id requires an unsigned integer"
          )
        }
        displayID = parsed
      case "--help", "-h":
        printUsage()
        Foundation.exit(EXIT_SUCCESS)
      default:
        throw RelayFailure.invalidSetup(
          "unknown argument \(arguments[index])"
        )
      }
      index += 1
    }
    guard setupJSON == nil || setupFile == nil else {
      throw RelayFailure.invalidSetup(
        "provide at most one of --setup or --setup-file"
      )
    }
    guard windowID == nil || displayID == nil else {
      throw RelayFailure.invalidSetup(
        "provide at most one of --window-id or --display-id"
      )
    }
    let setup: PairingSetup
    let setupFileURL: URL?
    let shouldPersistConfiguration: Bool
    if let setupJSON {
      setup = try decodePairingSetup(Data(setupJSON.utf8))
      setupFileURL = nil
      shouldPersistConfiguration = true
    } else if let setupFile {
      let fileURL = URL(fileURLWithPath: setupFile)
      setup = try decodePairingSetup(Data(contentsOf: fileURL))
      setupFileURL = fileURL
      shouldPersistConfiguration = true
    } else {
      let stored = try configurationStore.load()
      setup = PairingSetup(
        websocketURL: stored.websocketURL,
        pairingToken: nil
      )
      setupFileURL = nil
      shouldPersistConfiguration = false
    }
    return Self(
      setup: setup,
      windowID: windowID,
      displayID: displayID,
      configurationStore: configurationStore,
      setupFileURL: setupFileURL,
      shouldPersistConfiguration: shouldPersistConfiguration
    )
  }

  private static func decodePairingSetup(_ data: Data) throws -> PairingSetup {
    let setup = try JSONDecoder().decode(PairingSetup.self, from: data)
    guard
      let token = setup.pairingToken?
        .trimmingCharacters(in: .whitespacesAndNewlines),
      !token.isEmpty
    else {
      throw RelayFailure.invalidSetup(
        "one-time pairing setup must include pairing_token"
      )
    }
    return setup
  }
}

@MainActor
private final class RelayApplication {
  private let controller: DesktopController
  private let client: RelayClient
  private var eventMonitors: [Any] = []

  init(controller: DesktopController, client: RelayClient) {
    self.controller = controller
    self.client = client
  }

  func run() {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    installEmergencyControls()
    print(
      """
      Desktop Relay is running.
        Pause/resume: Command-Option-P
        Emergency stop: Command-Option-Escape
      Leave this process running while Xagent uses the authorized desktop target.
      """
    )
    Task.detached(priority: .userInitiated) { [client] in
      await client.runForever()
    }
    app.run()
  }

  private func installEmergencyControls() {
    let handler: (NSEvent) -> Void = { [weak self] event in
      guard Self.hasControlModifiers(event) else { return }
      if event.keyCode == 53 {
        Task { await self?.stopImmediately() }
      } else if event.charactersIgnoringModifiers?.lowercased() == "p" {
        Task { await self?.togglePause() }
      }
    }
    if let local = NSEvent.addLocalMonitorForEvents(
      matching: .keyDown,
      handler: { event in
        handler(event)
        return event
      }
    ) {
      eventMonitors.append(local)
    }
    if let global = NSEvent.addGlobalMonitorForEvents(
      matching: .keyDown,
      handler: handler
    ) {
      eventMonitors.append(global)
    }
  }

  private func togglePause() async {
    await controller.togglePause()
    let status = await controller.status()
    print(status.paused ? "Desktop Relay paused." : "Desktop Relay resumed.")
    await client.sendCurrentStatus()
  }

  private func stopImmediately() async {
    await controller.emergencyStop()
    print(
      "Desktop Relay emergency stop activated; the desktop target authorization was cleared."
    )
    await client.sendCurrentStatus()
    await client.stop()
    NSApplication.shared.terminate(nil)
  }

  private static func hasControlModifiers(_ event: NSEvent) -> Bool {
    let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
    return flags.contains(.command) && flags.contains(.option)
  }
}

@main
private enum DesktopRelayMain {
  @MainActor
  static func main() async {
    do {
      if CommandLine.arguments.dropFirst() == ["--self-test"] {
        try runSelfTest()
        print("Desktop Relay self-test passed.")
        return
      }
      let options = try CommandLineOptions.parse(CommandLine.arguments)
      let permissions = DesktopController.requestPermissions()
      if permissions["screen_recording"] != true {
        print(
          "Screen Recording permission was requested. Enable it for this binary in System Settings, then restart Desktop Relay."
        )
      }
      if permissions["accessibility"] != true {
        print(
          "Accessibility permission was requested. Enable it for this binary in System Settings, then restart Desktop Relay."
        )
      }

      let windows = try await DesktopController.availableWindows()
      let displays = try await DesktopController.availableDisplays()
      guard !windows.isEmpty || !displays.isEmpty else {
        throw RelayFailure.windowUnavailable(
          "no shareable displays or windows are available"
        )
      }
      let selectedTarget = try selectTarget(
        windows: windows,
        displays: displays,
        requestedWindowID: options.windowID,
        requestedDisplayID: options.displayID
      )
      let controller = DesktopController()
      switch selectedTarget {
      case .window(let windowID):
        try await controller.authorize(windowID: windowID)
      case .display(let displayID):
        try await controller.authorize(displayID: displayID)
      }
      let pairingCompleted: RelayClient.PairingCompleted?
      if options.shouldPersistConfiguration {
        let configurationStore = options.configurationStore
        let websocketURL = options.setup.websocketURL
        let setupFileURL = options.setupFileURL
        pairingCompleted = {
          try configurationStore.save(websocketURL: websocketURL)
          configurationStore.removeManagedPairingFileIfNeeded(setupFileURL)
          print(
            "Desktop Relay configuration saved to "
              + configurationStore.configurationURL.path
          )
        }
      } else {
        pairingCompleted = nil
      }
      let client = try RelayClient(
        setup: options.setup,
        pairingCompleted: pairingCompleted,
        commandHandler: { command, sendMediaChunk in
          try await controller.handle(
            command,
            sendMediaChunk: sendMediaChunk
          )
        },
        statusProvider: {
          await controller.status()
        }
      )
      RelayApplication(controller: controller, client: client).run()
    } catch {
      FileHandle.standardError.write(
        Data("Desktop Relay failed: \(error.localizedDescription)\n".utf8)
      )
      printUsage()
      Foundation.exit(EXIT_FAILURE)
    }
  }

  private enum SelectedDesktopTarget {
    case window(CGWindowID)
    case display(CGDirectDisplayID)
  }

  private static func selectTarget(
    windows: [WindowChoice],
    displays: [DisplayChoice],
    requestedWindowID: CGWindowID?,
    requestedDisplayID: CGDirectDisplayID?
  ) throws -> SelectedDesktopTarget {
    if let requestedWindowID {
      return .window(
        try selectWindow(
          windows,
          requestedWindowID: requestedWindowID
        )
      )
    }
    if let requestedDisplayID {
      return .display(
        try selectDisplay(
          displays,
          requestedDisplayID: requestedDisplayID
        )
      )
    }
    print("Choose what Xagent may observe and control:")
    print("  1. Entire display (switch between apps and windows)")
    print("  2. One window only")
    print("Authorization scope: ", terminator: "")
    guard let input = readLine(), let selection = Int(input) else {
      throw RelayFailure.invalidSetup("invalid authorization scope")
    }
    switch selection {
    case 1:
      return .display(
        try selectDisplay(displays, requestedDisplayID: nil)
      )
    case 2:
      return .window(
        try selectWindow(windows, requestedWindowID: nil)
      )
    default:
      throw RelayFailure.invalidSetup("invalid authorization scope")
    }
  }

  private static func runSelfTest() throws {
    let data = Data(
      #"{"websocket_url":"wss://example.test/ws/desktop-relay","pairing_token":"once"}"#.utf8
    )
    let setup = try JSONDecoder().decode(PairingSetup.self, from: data)
    guard
      setup.websocketURL == "wss://example.test/ws/desktop-relay",
      setup.pairingToken == "once"
    else {
      throw RelayFailure.invalidSetup("pairing setup decoding failed")
    }
    var rejectedOutOfBoundsPoint = false
    do {
      _ = try ComputerPoint(
        json: ["x": .number(2), "y": .number(0.5)]
      )
    } catch RelayFailure.invalidAction {
      rejectedOutOfBoundsPoint = true
    }
    guard rejectedOutOfBoundsPoint else {
      throw RelayFailure.invalidAction(
        "out-of-window coordinates were accepted"
      )
    }

    let stableSample: [UInt8] = [0, 64, 128, 255]
    let changedSample: [UInt8] = [255, 128, 64, 0]
    var detector = VisualStabilityDetector(
      maximumNormalizedDifference: 0.003,
      requiredStableComparisons: 2
    )
    guard
      !detector.observe(stableSample),
      !detector.observe(stableSample),
      detector.observe(stableSample),
      !detector.observe(changedSample),
      !detector.observe(changedSample),
      detector.observe(changedSample)
    else {
      throw RelayFailure.invalidAction(
        "visual stability detector did not settle or reset as expected"
      )
    }

    let temporaryDirectory = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let configurationStore = DesktopRelayConfigurationStore(
      directoryURL: temporaryDirectory.appendingPathComponent(
        "desktop-relay",
        isDirectory: true
      )
    )
    try configurationStore.save(
      websocketURL: "wss://example.test/ws/desktop-relay"
    )
    let stored = try configurationStore.load()
    guard stored.websocketURL == "wss://example.test/ws/desktop-relay" else {
      throw RelayFailure.invalidSetup("stored relay endpoint did not round-trip")
    }
    let persistedJSON = try String(
      contentsOf: configurationStore.configurationURL,
      encoding: .utf8
    )
    guard !persistedJSON.contains("pairing_token") else {
      throw RelayFailure.invalidSetup(
        "one-time pairing token leaked into persistent configuration"
      )
    }
    let attributes = try FileManager.default.attributesOfItem(
      atPath: configurationStore.configurationURL.path
    )
    guard
      (attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600
    else {
      throw RelayFailure.invalidSetup(
        "persistent relay configuration permissions are not 0600"
      )
    }
    let storedOptions = try CommandLineOptions.parse(
      ["xagent-desktop-relay"],
      configurationStore: configurationStore
    )
    guard
      storedOptions.setup.websocketURL
        == "wss://example.test/ws/desktop-relay",
      storedOptions.setup.pairingToken == nil,
      !storedOptions.shouldPersistConfiguration
    else {
      throw RelayFailure.invalidSetup(
        "no-argument launch did not use stored relay configuration"
      )
    }
    let displayOptions = try CommandLineOptions.parse(
      ["xagent-desktop-relay", "--display-id", "5"],
      configurationStore: configurationStore
    )
    guard displayOptions.displayID == 5, displayOptions.windowID == nil else {
      throw RelayFailure.invalidSetup(
        "display authorization argument did not parse"
      )
    }
    let managedPairingURL = configurationStore.directoryURL
      .appendingPathComponent("pairing.json", isDirectory: false)
    try Data("one-time".utf8).write(to: managedPairingURL)
    configurationStore.removeManagedPairingFileIfNeeded(managedPairingURL)
    guard !FileManager.default.fileExists(atPath: managedPairingURL.path) else {
      throw RelayFailure.invalidSetup(
        "managed one-time pairing file was not removed"
      )
    }
  }

  private static func selectWindow(
    _ windows: [WindowChoice],
    requestedWindowID: CGWindowID?
  ) throws -> CGWindowID {
    if let requestedWindowID {
      guard windows.contains(where: { $0.windowID == requestedWindowID }) else {
        throw RelayFailure.windowUnavailable(
          "window \(requestedWindowID) is not currently shareable"
        )
      }
      return requestedWindowID
    }
    print("Select the single window Xagent may observe and control:")
    for (index, window) in windows.enumerated() {
      print(
        "  \(index + 1). \(window.application) — \(window.title) [\(window.windowID)]"
      )
    }
    print("Window number: ", terminator: "")
    guard
      let input = readLine(),
      let selection = Int(input),
      windows.indices.contains(selection - 1)
    else {
      throw RelayFailure.invalidSetup("invalid window selection")
    }
    return windows[selection - 1].windowID
  }

  private static func selectDisplay(
    _ displays: [DisplayChoice],
    requestedDisplayID: CGDirectDisplayID?
  ) throws -> CGDirectDisplayID {
    if let requestedDisplayID {
      guard displays.contains(where: { $0.displayID == requestedDisplayID })
      else {
        throw RelayFailure.windowUnavailable(
          "display \(requestedDisplayID) is not currently shareable"
        )
      }
      return requestedDisplayID
    }
    guard !displays.isEmpty else {
      throw RelayFailure.windowUnavailable(
        "no shareable displays are available"
      )
    }
    print("Select the display Xagent may observe and control:")
    for (index, display) in displays.enumerated() {
      print(
        "  \(index + 1). \(display.name) — "
          + "\(Int(display.frame.width))×\(Int(display.frame.height)) "
          + "[\(display.displayID)]"
      )
    }
    print("Display number: ", terminator: "")
    guard
      let input = readLine(),
      let selection = Int(input),
      displays.indices.contains(selection - 1)
    else {
      throw RelayFailure.invalidSetup("invalid display selection")
    }
    return displays[selection - 1].displayID
  }
}

private func printUsage() {
  print(
    """
    Usage:
      xagent-desktop-relay [--window-id ID | --display-id ID]
      xagent-desktop-relay --setup '<pairing-json>' [--window-id ID | --display-id ID]
      xagent-desktop-relay --setup-file PATH [--window-id ID | --display-id ID]
      xagent-desktop-relay --self-test

    Create the one-time pairing JSON in Xagent Settings > Computer Use.
    After the first successful pairing, the server address is stored under
    $XAGENT_STORAGE_ROOT/desktop-relay (default: ~/.xagent/desktop-relay)
    and later launches need no setup argument. Session credentials stay in
    macOS Keychain.
    Without a target ID, the relay asks whether to authorize one entire display
    or one specific window.
    """
  )
}
