import AppKit
import ApplicationServices
import CoreGraphics
import Foundation
import ScreenCaptureKit

struct WindowChoice: Sendable {
  let windowID: CGWindowID
  let application: String
  let title: String
  let frame: CGRect
}

private struct AuthorizedWindow {
  let windowID: CGWindowID
  let processID: pid_t
  let application: String
  let title: String
}

struct VisualStabilityDetector {
  private let maximumNormalizedDifference: Double
  private let requiredStableComparisons: Int
  private var previousSample: [UInt8]?
  private var stableComparisons = 0

  init(
    maximumNormalizedDifference: Double = 0.003,
    requiredStableComparisons: Int = 2
  ) {
    self.maximumNormalizedDifference = maximumNormalizedDifference
    self.requiredStableComparisons = max(1, requiredStableComparisons)
  }

  mutating func observe(_ sample: [UInt8]) -> Bool {
    guard !sample.isEmpty else {
      previousSample = nil
      stableComparisons = 0
      return false
    }
    guard let previousSample, previousSample.count == sample.count else {
      self.previousSample = sample
      stableComparisons = 0
      return false
    }

    let totalDifference = zip(previousSample, sample).reduce(0.0) { partial, pair in
      partial + abs(Double(pair.0) - Double(pair.1))
    }
    let normalizedDifference =
      totalDifference / (Double(sample.count) * Double(UInt8.max))
    stableComparisons =
      normalizedDifference <= maximumNormalizedDifference
      ? stableComparisons + 1
      : 0
    self.previousSample = sample
    return stableComparisons >= requiredStableComparisons
  }
}

actor DesktopController {
  private static let visualSampleSize = 64
  private static let visualSettleInitialDelay = Duration.milliseconds(300)
  private static let visualSettleInterval = Duration.milliseconds(200)
  private static let visualSettleMaximumSamples = 8

  private var authorized: AuthorizedWindow?
  private var paused = false
  private var emergencyStopped = false
  private var lastFrameID: String?
  private var lastWindowFrame: CGRect?
  private var lastElements: [String: AXUIElement] = [:]
  private var sensitiveElementIDs: Set<String> = []

  static func requestPermissions() -> [String: Bool] {
    let screenRecording =
      CGPreflightScreenCaptureAccess()
      || CGRequestScreenCaptureAccess()
    let prompt = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
    let accessibility = AXIsProcessTrustedWithOptions(prompt)
    return [
      "screen_recording": screenRecording,
      "accessibility": accessibility,
    ]
  }

  static func availableWindows() async throws -> [WindowChoice] {
    let content = try await SCShareableContent.excludingDesktopWindows(
      true,
      onScreenWindowsOnly: true
    )
    let ownPID = ProcessInfo.processInfo.processIdentifier
    return content.windows.compactMap { window in
      guard
        window.owningApplication?.processID != ownPID,
        window.frame.width >= 80,
        window.frame.height >= 60,
        window.isOnScreen
      else {
        return nil
      }
      return WindowChoice(
        windowID: window.windowID,
        application: window.owningApplication?.applicationName ?? "Unknown",
        title: window.title ?? "Untitled",
        frame: window.frame
      )
    }.sorted {
      ($0.application, $0.title, $0.windowID)
        < ($1.application, $1.title, $1.windowID)
    }
  }

  func authorize(windowID: CGWindowID) async throws {
    let window = try await currentShareableWindow(windowID: windowID)
    guard let application = window.owningApplication else {
      throw RelayFailure.windowUnavailable("selected window has no owning application")
    }
    authorized = AuthorizedWindow(
      windowID: window.windowID,
      processID: application.processID,
      application: application.applicationName,
      title: window.title ?? "Untitled"
    )
    paused = false
    emergencyStopped = false
    lastFrameID = nil
    lastWindowFrame = nil
    lastElements.removeAll()
    sensitiveElementIDs.removeAll()
  }

  func togglePause() {
    guard authorized != nil, !emergencyStopped else { return }
    paused.toggle()
    lastFrameID = nil
    lastWindowFrame = nil
    lastElements.removeAll()
    sensitiveElementIDs.removeAll()
  }

  func emergencyStop() {
    emergencyStopped = true
    paused = true
    authorized = nil
    lastFrameID = nil
    lastWindowFrame = nil
    lastElements.removeAll()
    sensitiveElementIDs.removeAll()
  }

  func status() async -> DesktopWindowStatus {
    let permissions = Self.permissionStatus()
    guard let authorized else {
      return DesktopWindowStatus(
        attached: false,
        windowID: nil,
        title: nil,
        application: nil,
        bounds: nil,
        permissions: permissions,
        paused: paused,
        emergencyStopped: emergencyStopped
      )
    }
    let window = try? await validateAuthorizedWindow()
    return DesktopWindowStatus(
      attached: window != nil,
      windowID: authorized.windowID,
      title: authorized.title,
      application: authorized.application,
      bounds: window.map {
        WindowBounds(
          x: $0.frame.origin.x,
          y: $0.frame.origin.y,
          width: $0.frame.width,
          height: $0.frame.height
        )
      },
      permissions: permissions,
      paused: paused,
      emergencyStopped: emergencyStopped
    )
  }

  func handle(_ command: RelayCommand) async throws -> [String: JSONValue] {
    switch command.command {
    case "observe":
      let frameID = try requiredString(command.payload["frame_id"], "frame_id")
      let observation = try await observe(frameID: frameID)
      return ["observation": try jsonValue(observation)]
    case "act":
      guard !emergencyStopped else { throw RelayFailure.emergencyStopped }
      guard !paused else { throw RelayFailure.paused }
      let expected = try requiredString(
        command.payload["expected_frame_id"],
        "expected_frame_id"
      )
      guard expected == lastFrameID else {
        throw RelayFailure.invalidAction(
          "the authorized window changed after the last screenshot; request a fresh screenshot"
        )
      }
      let frameID = try requiredString(command.payload["frame_id"], "frame_id")
      guard let action = command.payload["action"]?.objectValue else {
        throw RelayFailure.invalidAction("action must be an object")
      }
      let shouldSettle = try await perform(action)
      if shouldSettle {
        try await waitForVisualStability()
      }
      let observation = try await observe(frameID: frameID)
      return ["observation": try jsonValue(observation)]
    default:
      throw RelayFailure.invalidMessage(
        "unsupported desktop relay command \(command.command)"
      )
    }
  }

  private func observe(frameID: String) async throws -> ObservationPayload {
    guard !emergencyStopped else { throw RelayFailure.emergencyStopped }
    let permissions = Self.permissionStatus()
    guard permissions["screen_recording"] == true else {
      throw RelayFailure.permission(
        "Screen Recording permission is required in System Settings"
      )
    }
    let window = try await validateAuthorizedWindow()
    let scale = Self.backingScale(for: window.frame)
    let configuration = SCStreamConfiguration()
    configuration.width = max(1, Int(window.frame.width * scale))
    configuration.height = max(1, Int(window.frame.height * scale))
    configuration.showsCursor = true
    configuration.ignoreShadowsSingleWindow = true
    let filter = SCContentFilter(desktopIndependentWindow: window)
    let image = try await SCScreenshotManager.captureImage(
      contentFilter: filter,
      configuration: configuration
    )
    let representation = NSBitmapImageRep(cgImage: image)
    guard let png = representation.representation(using: .png, properties: [:]) else {
      throw RelayFailure.invalidMessage("could not encode desktop screenshot")
    }

    let extraction = Self.extractElements(
      processID: authorized?.processID,
      window: window,
      accessibilityAllowed: permissions["accessibility"] == true
    )
    lastElements = extraction.handles
    sensitiveElementIDs = extraction.sensitiveIDs
    lastFrameID = frameID
    lastWindowFrame = window.frame
    return ObservationPayload(
      screenshotBase64: png.base64EncodedString(),
      viewport: ViewportPayload(
        width: max(1, Int(window.frame.width)),
        height: max(1, Int(window.frame.height)),
        devicePixelRatio: scale
      ),
      elements: extraction.elements,
      elementsTruncated: extraction.truncated,
      elementExtractionFailed: extraction.failed,
      elementExtractionIncomplete: extraction.incomplete,
      windowID: window.windowID,
      title: authorized?.title,
      application: authorized?.application,
      paused: paused,
      emergencyStopped: emergencyStopped
    )
  }

  private func perform(_ action: [String: JSONValue]) async throws -> Bool {
    guard Self.permissionStatus()["accessibility"] == true else {
      throw RelayFailure.permission(
        "Accessibility permission is required in System Settings"
      )
    }
    let type = try requiredString(action["type"], "action type")
    if type == "screenshot" { return false }
    if type == "navigate" {
      throw RelayFailure.invalidAction("navigate is not supported on desktop")
    }
    if type == "wait" {
      let duration = max(0, min(30_000, Int(action["duration_ms"]?.numberValue ?? 1_000)))
      try await Task.sleep(for: .milliseconds(duration))
      return false
    }

    let window = try await validateAuthorizedWindow()
    guard Self.sameFrame(window.frame, lastWindowFrame) else {
      throw RelayFailure.invalidAction(
        "the authorized window moved or resized after the last screenshot; request a fresh screenshot"
      )
    }
    let target = try action["target"]?.objectValue.map(ComputerPoint.init(json:))
    let targetID = action["target_element_id"]?.stringValue
    let point = target.map { Self.globalPoint($0, in: window.frame) }

    if let point {
      try verifyHit(targetID: targetID, at: point, window: window)
    }
    switch type {
    case "click", "double_click":
      guard let point else {
        throw RelayFailure.invalidAction("\(type) requires a target")
      }
      Self.click(at: point, count: type == "double_click" ? 2 : 1)
      return true
    case "move":
      guard let point else {
        throw RelayFailure.invalidAction("move requires a target")
      }
      Self.postMouse(type: .mouseMoved, at: point)
      return false
    case "type":
      if let targetID, sensitiveElementIDs.contains(targetID) {
        throw RelayFailure.invalidAction(
          "credentials and secure fields must be entered by the user"
        )
      }
      if let point {
        Self.click(at: point, count: 1)
      } else {
        try verifyAuthorizedWindowIsFocused(window: window)
      }
      try Self.typeText(action["text"]?.stringValue ?? "")
      return true
    case "keypress":
      try verifyAuthorizedWindowIsFocused(window: window)
      let keys =
        action["keys"].flatMap { value -> [String]? in
          guard case .array(let items) = value else { return nil }
          return items.compactMap(\.stringValue)
        } ?? []
      try Self.press(keys: keys)
      return true
    case "scroll":
      let deltaX = action["delta_x"]?.numberValue ?? 0
      let deltaY = action["delta_y"]?.numberValue ?? 0
      let scrollPoint =
        point
        ?? CGPoint(
          x: window.frame.midX,
          y: window.frame.midY
        )
      if point == nil {
        try verifyHit(targetID: nil, at: scrollPoint, window: window)
      }
      Self.scroll(
        x: deltaX * window.frame.width,
        y: deltaY * window.frame.height,
        at: scrollPoint
      )
      return true
    case "drag":
      guard
        let startJSON = action["start"]?.objectValue,
        let endJSON = action["end"]?.objectValue
      else {
        throw RelayFailure.invalidAction("drag requires start and end")
      }
      let start = Self.globalPoint(try ComputerPoint(json: startJSON), in: window.frame)
      let end = Self.globalPoint(try ComputerPoint(json: endJSON), in: window.frame)
      try verifyHit(targetID: nil, at: start, window: window)
      try verifyHit(targetID: nil, at: end, window: window)
      try verifyHit(
        targetID: nil,
        at: CGPoint(x: (start.x + end.x) / 2, y: (start.y + end.y) / 2),
        window: window
      )
      let duration = max(0, min(30_000, Int(action["duration_ms"]?.numberValue ?? 250)))
      try await Self.drag(from: start, to: end, durationMilliseconds: duration)
      return true
    default:
      throw RelayFailure.invalidAction("unsupported desktop action \(type)")
    }
  }

  private func waitForVisualStability() async throws {
    try await Task.sleep(for: Self.visualSettleInitialDelay)
    var detector = VisualStabilityDetector()

    for index in 0..<Self.visualSettleMaximumSamples {
      do {
        let window = try await validateAuthorizedWindow()
        let sample = try await captureVisualSample(window: window)
        if detector.observe(sample) {
          return
        }
      } catch {
        if Task.isCancelled {
          throw error
        }
        // Stabilization is best-effort. The full observation below remains the
        // source of truth and will surface any persistent capture failure.
        return
      }
      if index + 1 < Self.visualSettleMaximumSamples {
        try await Task.sleep(for: Self.visualSettleInterval)
      }
    }
  }

  private func captureVisualSample(window: SCWindow) async throws -> [UInt8] {
    let configuration = SCStreamConfiguration()
    configuration.width = Self.visualSampleSize
    configuration.height = Self.visualSampleSize
    configuration.showsCursor = false
    configuration.ignoreShadowsSingleWindow = true
    let filter = SCContentFilter(desktopIndependentWindow: window)
    let image = try await SCScreenshotManager.captureImage(
      contentFilter: filter,
      configuration: configuration
    )

    var pixels = [UInt8](
      repeating: 0,
      count: Self.visualSampleSize * Self.visualSampleSize
    )
    let rendered = pixels.withUnsafeMutableBytes { bytes -> Bool in
      guard
        let baseAddress = bytes.baseAddress,
        let context = CGContext(
          data: baseAddress,
          width: Self.visualSampleSize,
          height: Self.visualSampleSize,
          bitsPerComponent: 8,
          bytesPerRow: Self.visualSampleSize,
          space: CGColorSpaceCreateDeviceGray(),
          bitmapInfo: CGImageAlphaInfo.none.rawValue
        )
      else {
        return false
      }
      context.interpolationQuality = .low
      context.draw(
        image,
        in: CGRect(
          x: 0,
          y: 0,
          width: Self.visualSampleSize,
          height: Self.visualSampleSize
        )
      )
      return true
    }
    guard rendered else {
      throw RelayFailure.invalidMessage(
        "could not sample the authorized desktop window"
      )
    }
    return pixels
  }

  private func verifyHit(
    targetID: String?,
    at point: CGPoint,
    window: SCWindow
  ) throws {
    guard let authorized else {
      throw RelayFailure.invalidAction(
        "desktop window authorization is unavailable"
      )
    }
    let expected = targetID.flatMap { lastElements[$0] }
    if targetID != nil, expected == nil {
      throw RelayFailure.invalidAction(
        "target identity is unavailable; request a fresh screenshot"
      )
    }
    let application = AXUIElementCreateApplication(authorized.processID)
    guard
      let selectedWindow = Self.matchingAXWindow(
        application: application,
        frame: window.frame
      )
    else {
      throw RelayFailure.invalidAction(
        "could not verify the authorized desktop window"
      )
    }
    let system = AXUIElementCreateSystemWide()
    var hit: AXUIElement?
    guard
      AXUIElementCopyElementAtPosition(
        system,
        Float(point.x),
        Float(point.y),
        &hit
      ) == .success,
      var candidate = hit
    else {
      throw RelayFailure.invalidAction("could not verify the desktop target")
    }
    var processID: pid_t = 0
    guard
      AXUIElementGetPid(candidate, &processID) == .success,
      processID == authorized.processID
    else {
      throw RelayFailure.invalidAction(
        "another application covers the authorized desktop window"
      )
    }
    var foundExpected = expected == nil
    var foundWindow = false
    for _ in 0..<32 {
      if let expected, CFEqual(candidate, expected) {
        foundExpected = true
      }
      if CFEqual(candidate, selectedWindow) {
        foundWindow = true
        break
      }
      guard let parent = Self.axElement(candidate, kAXParentAttribute) else { break }
      candidate = parent
    }
    if foundExpected, foundWindow { return }
    throw RelayFailure.invalidAction(
      "the selected desktop control is covered or changed; request a fresh screenshot"
    )
  }

  private func verifyAuthorizedWindowIsFocused(window: SCWindow) throws {
    guard let authorized else {
      throw RelayFailure.invalidAction(
        "desktop window authorization is unavailable"
      )
    }
    let application = AXUIElementCreateApplication(authorized.processID)
    guard
      let selectedWindow = Self.matchingAXWindow(
        application: application,
        frame: window.frame
      ),
      let focusedWindow = Self.axElement(
        application,
        kAXFocusedWindowAttribute
      ),
      CFEqual(selectedWindow, focusedWindow)
    else {
      throw RelayFailure.invalidAction(
        "the authorized desktop window is not focused; request a fresh screenshot and click it before sending keyboard input"
      )
    }
  }

  private func validateAuthorizedWindow() async throws -> SCWindow {
    guard let authorized else {
      throw RelayFailure.windowUnavailable(
        "no desktop window is authorized; select one in Desktop Relay"
      )
    }
    let window = try await currentShareableWindow(windowID: authorized.windowID)
    guard window.owningApplication?.processID == authorized.processID else {
      throw RelayFailure.windowUnavailable(
        "the authorized window was closed or replaced"
      )
    }
    return window
  }

  private func currentShareableWindow(windowID: CGWindowID) async throws -> SCWindow {
    let content = try await SCShareableContent.excludingDesktopWindows(
      true,
      onScreenWindowsOnly: true
    )
    guard let window = content.windows.first(where: { $0.windowID == windowID }) else {
      throw RelayFailure.windowUnavailable("the selected window is no longer available")
    }
    return window
  }

  private static func permissionStatus() -> [String: Bool] {
    [
      "screen_recording": CGPreflightScreenCaptureAccess(),
      "accessibility": AXIsProcessTrusted(),
    ]
  }

  private static func backingScale(for frame: CGRect) -> Double {
    for screen in NSScreen.screens {
      guard
        let number = screen.deviceDescription[
          NSDeviceDescriptionKey("NSScreenNumber")
        ] as? NSNumber
      else {
        continue
      }
      let displayBounds = CGDisplayBounds(
        CGDirectDisplayID(number.uint32Value)
      )
      if displayBounds.intersects(frame) {
        return max(1, screen.backingScaleFactor)
      }
    }
    return 1
  }

  private static func globalPoint(_ point: ComputerPoint, in frame: CGRect) -> CGPoint {
    CGPoint(
      x: frame.minX + point.x * frame.width,
      y: frame.minY + point.y * frame.height
    )
  }

  private static func sameFrame(_ current: CGRect, _ previous: CGRect?) -> Bool {
    guard let previous else { return false }
    let tolerance = 0.5
    return abs(current.minX - previous.minX) <= tolerance
      && abs(current.minY - previous.minY) <= tolerance
      && abs(current.width - previous.width) <= tolerance
      && abs(current.height - previous.height) <= tolerance
  }

  private static func click(at point: CGPoint, count: Int) {
    for index in 1...count {
      postMouse(type: .leftMouseDown, at: point, clickState: Int64(index))
      postMouse(type: .leftMouseUp, at: point, clickState: Int64(index))
    }
  }

  private static func postMouse(
    type: CGEventType,
    at point: CGPoint,
    clickState: Int64 = 1
  ) {
    guard
      let event = CGEvent(
        mouseEventSource: nil,
        mouseType: type,
        mouseCursorPosition: point,
        mouseButton: .left
      )
    else { return }
    event.setIntegerValueField(.mouseEventClickState, value: clickState)
    event.post(tap: .cghidEventTap)
  }

  private static func scroll(x: Double, y: Double, at point: CGPoint) {
    guard
      let event = CGEvent(
        scrollWheelEvent2Source: nil,
        units: .pixel,
        wheelCount: 2,
        wheel1: Int32(-y),
        wheel2: Int32(-x),
        wheel3: 0
      )
    else { return }
    event.location = point
    event.post(tap: .cghidEventTap)
  }

  private static func drag(
    from start: CGPoint,
    to end: CGPoint,
    durationMilliseconds: Int
  ) async throws {
    postMouse(type: .leftMouseDown, at: start)
    let steps = max(1, min(50, durationMilliseconds / 25))
    for step in 1...steps {
      let progress = Double(step) / Double(steps)
      let point = CGPoint(
        x: start.x + (end.x - start.x) * progress,
        y: start.y + (end.y - start.y) * progress
      )
      postMouse(type: .leftMouseDragged, at: point)
      if durationMilliseconds > 0 {
        try await Task.sleep(
          for: .milliseconds(max(1, durationMilliseconds / steps))
        )
      }
    }
    postMouse(type: .leftMouseUp, at: end)
  }

  private static func typeText(_ text: String) throws {
    var characters = Array(text.utf16)
    guard
      let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true),
      let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false)
    else {
      throw RelayFailure.invalidAction("could not create keyboard event")
    }
    characters.withUnsafeMutableBufferPointer { buffer in
      down.keyboardSetUnicodeString(
        stringLength: buffer.count,
        unicodeString: buffer.baseAddress
      )
      up.keyboardSetUnicodeString(
        stringLength: buffer.count,
        unicodeString: buffer.baseAddress
      )
    }
    down.post(tap: .cghidEventTap)
    up.post(tap: .cghidEventTap)
  }

  private static func press(keys: [String]) throws {
    let normalized = keys.map { $0.uppercased() }
    var flags: CGEventFlags = []
    if normalized.contains("SHIFT") { flags.insert(.maskShift) }
    if normalized.contains("CTRL") || normalized.contains("CONTROL") {
      flags.insert(.maskControl)
    }
    if normalized.contains("ALT") || normalized.contains("OPTION") {
      flags.insert(.maskAlternate)
    }
    if normalized.contains("CMD") || normalized.contains("COMMAND")
      || normalized.contains("META")
    {
      flags.insert(.maskCommand)
    }
    let modifiers = Set([
      "SHIFT", "CTRL", "CONTROL", "ALT", "OPTION", "CMD", "COMMAND", "META",
    ])
    guard
      let key = normalized.first(where: { !modifiers.contains($0) }),
      let code = keyCode(key)
    else {
      throw RelayFailure.invalidAction("keypress requires one supported key")
    }
    guard
      let down = CGEvent(keyboardEventSource: nil, virtualKey: code, keyDown: true),
      let up = CGEvent(keyboardEventSource: nil, virtualKey: code, keyDown: false)
    else {
      throw RelayFailure.invalidAction("could not create keypress event")
    }
    down.flags = flags
    up.flags = flags
    down.post(tap: .cghidEventTap)
    up.post(tap: .cghidEventTap)
  }

  private static func keyCode(_ key: String) -> CGKeyCode? {
    let codes: [String: CGKeyCode] = [
      "A": 0, "S": 1, "D": 2, "F": 3, "H": 4, "G": 5, "Z": 6,
      "X": 7, "C": 8, "V": 9, "B": 11, "Q": 12, "W": 13, "E": 14,
      "R": 15, "Y": 16, "T": 17, "1": 18, "2": 19, "3": 20, "4": 21,
      "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29, "O": 31,
      "U": 32, "I": 34, "P": 35, "L": 37, "J": 38, "K": 40, "N": 45,
      "M": 46, "RETURN": 36, "ENTER": 36, "TAB": 48, "SPACE": 49,
      "BACKSPACE": 51, "ESC": 53, "ESCAPE": 53, "LEFT": 123,
      "RIGHT": 124, "DOWN": 125, "UP": 126,
    ]
    return codes[key]
  }
}

extension DesktopController {
  fileprivate struct AXExtraction {
    let elements: [ElementPayload]
    let handles: [String: AXUIElement]
    let sensitiveIDs: Set<String>
    let truncated: Bool
    let failed: Bool
    let incomplete: Bool
  }

  fileprivate static func extractElements(
    processID: pid_t?,
    window: SCWindow,
    accessibilityAllowed: Bool
  ) -> AXExtraction {
    guard accessibilityAllowed, let processID else {
      return AXExtraction(
        elements: [],
        handles: [:],
        sensitiveIDs: [],
        truncated: false,
        failed: !accessibilityAllowed,
        incomplete: true
      )
    }
    let application = AXUIElementCreateApplication(processID)
    guard let root = matchingAXWindow(application: application, frame: window.frame) else {
      return AXExtraction(
        elements: [],
        handles: [:],
        sensitiveIDs: [],
        truncated: false,
        failed: true,
        incomplete: true
      )
    }
    let actionable = Set([
      kAXButtonRole, kAXTextFieldRole, kAXTextAreaRole, kAXCheckBoxRole,
      kAXRadioButtonRole, "AXLink", kAXPopUpButtonRole, kAXMenuItemRole,
      kAXSliderRole, kAXComboBoxRole,
    ])
    var queue: [(AXUIElement, Int)] = [(root, 0)]
    var elements: [ElementPayload] = []
    var handles: [String: AXUIElement] = [:]
    var sensitiveIDs: Set<String> = []
    var visited = 0
    let maxVisited = 1_500
    let maxElements = 100

    while !queue.isEmpty, visited < maxVisited, elements.count < maxElements {
      let (element, depth) = queue.removeFirst()
      visited += 1
      let role = axString(element, kAXRoleAttribute)
      if let role, actionable.contains(role),
        let frame = axFrame(element),
        let normalized = normalizedBounds(frame, within: window.frame)
      {
        let elementID = "ax-\(elements.count + 1)"
        let subrole = axString(element, kAXSubroleAttribute)
        let sensitive =
          subrole == kAXSecureTextFieldSubrole
          || role.localizedCaseInsensitiveContains("secure")
        let label = firstNonempty([
          axString(element, kAXTitleAttribute),
          axString(element, kAXDescriptionAttribute),
          axString(element, kAXHelpAttribute),
        ])
        let focused = axBool(element, kAXFocusedAttribute)
        let enabled = axBool(element, kAXEnabledAttribute)
        elements.append(
          ElementPayload(
            elementID: elementID,
            bounds: normalized,
            label: sensitive ? "Sensitive input" : label,
            role: role,
            text: nil,
            metadata: [
              "role": .string(role),
              "subrole": subrole.map(JSONValue.string) ?? .null,
              "focused": .bool(focused ?? false),
              "enabled": .bool(enabled ?? true),
              "sensitive": .bool(sensitive),
            ]
          )
        )
        handles[elementID] = element
        if sensitive { sensitiveIDs.insert(elementID) }
      }
      if depth < 40, let children = axElements(element, kAXChildrenAttribute) {
        queue.append(contentsOf: children.map { ($0, depth + 1) })
      }
    }
    let truncated = !queue.isEmpty || visited >= maxVisited
    return AXExtraction(
      elements: elements,
      handles: handles,
      sensitiveIDs: sensitiveIDs,
      truncated: truncated,
      failed: false,
      incomplete: truncated
    )
  }

  fileprivate static func matchingAXWindow(
    application: AXUIElement,
    frame: CGRect
  ) -> AXUIElement? {
    guard let windows = axElements(application, kAXWindowsAttribute) else {
      return nil
    }
    return windows.min { first, second in
      frameDistance(axFrame(first), frame) < frameDistance(axFrame(second), frame)
    }
  }

  fileprivate static func frameDistance(_ candidate: CGRect?, _ expected: CGRect) -> Double {
    guard let candidate else { return .infinity }
    return abs(candidate.minX - expected.minX)
      + abs(candidate.minY - expected.minY)
      + abs(candidate.width - expected.width)
      + abs(candidate.height - expected.height)
  }

  fileprivate static func normalizedBounds(
    _ candidate: CGRect,
    within window: CGRect
  ) -> NormalizedBounds? {
    let clipped = candidate.intersection(window)
    guard !clipped.isNull, clipped.width >= 2, clipped.height >= 2 else {
      return nil
    }
    return NormalizedBounds(
      x: max(0, min(1, (clipped.minX - window.minX) / window.width)),
      y: max(0, min(1, (clipped.minY - window.minY) / window.height)),
      width: max(0.000_001, min(1, clipped.width / window.width)),
      height: max(0.000_001, min(1, clipped.height / window.height))
    )
  }

  fileprivate static func axValue(_ element: AXUIElement, _ attribute: String) -> CFTypeRef? {
    var value: CFTypeRef?
    guard
      AXUIElementCopyAttributeValue(
        element,
        attribute as CFString,
        &value
      ) == .success
    else {
      return nil
    }
    return value
  }

  fileprivate static func axString(_ element: AXUIElement, _ attribute: String) -> String? {
    axValue(element, attribute) as? String
  }

  fileprivate static func axBool(_ element: AXUIElement, _ attribute: String) -> Bool? {
    axValue(element, attribute) as? Bool
  }

  fileprivate static func axElements(
    _ element: AXUIElement,
    _ attribute: String
  ) -> [AXUIElement]? {
    axValue(element, attribute) as? [AXUIElement]
  }

  fileprivate static func axElement(
    _ element: AXUIElement,
    _ attribute: String
  ) -> AXUIElement? {
    guard let value = axValue(element, attribute),
      CFGetTypeID(value) == AXUIElementGetTypeID()
    else {
      return nil
    }
    return unsafeDowncast(value, to: AXUIElement.self)
  }

  fileprivate static func axFrame(_ element: AXUIElement) -> CGRect? {
    guard
      let positionValue = axValue(element, kAXPositionAttribute),
      let sizeValue = axValue(element, kAXSizeAttribute),
      CFGetTypeID(positionValue) == AXValueGetTypeID(),
      CFGetTypeID(sizeValue) == AXValueGetTypeID()
    else {
      return nil
    }
    var position = CGPoint.zero
    var size = CGSize.zero
    guard
      AXValueGetValue(positionValue as! AXValue, .cgPoint, &position),
      AXValueGetValue(sizeValue as! AXValue, .cgSize, &size)
    else {
      return nil
    }
    return CGRect(origin: position, size: size)
  }

  fileprivate static func firstNonempty(_ values: [String?]) -> String? {
    values.compactMap { value in
      let normalized = value?.trimmingCharacters(in: .whitespacesAndNewlines)
      return normalized?.isEmpty == false ? normalized : nil
    }.first.map { String($0.prefix(500)) }
  }
}

private func requiredString(_ value: JSONValue?, _ field: String) throws -> String {
  guard
    let normalized = value?.stringValue?.trimmingCharacters(
      in: .whitespacesAndNewlines
    ), !normalized.isEmpty
  else {
    throw RelayFailure.invalidMessage("\(field) must be a non-empty string")
  }
  return normalized
}
