import Foundation

struct StoredRelayConfiguration: Codable, Equatable, Sendable {
  let websocketURL: String

  enum CodingKeys: String, CodingKey {
    case websocketURL = "websocket_url"
  }
}

struct DesktopRelayConfigurationStore: Sendable {
  static let storageRootEnvironmentVariable = "XAGENT_STORAGE_ROOT"

  let directoryURL: URL

  var configurationURL: URL {
    directoryURL.appendingPathComponent("config.json", isDirectory: false)
  }

  static func defaultStore(
    environment: [String: String] = ProcessInfo.processInfo.environment,
    homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
  ) -> Self {
    let configuredRoot = environment[storageRootEnvironmentVariable]?
      .trimmingCharacters(in: .whitespacesAndNewlines)
    let storageRoot: URL
    if let configuredRoot, !configuredRoot.isEmpty {
      storageRoot = URL(
        fileURLWithPath: NSString(string: configuredRoot).expandingTildeInPath,
        isDirectory: true
      )
    } else {
      storageRoot = homeDirectory.appendingPathComponent(
        ".xagent",
        isDirectory: true
      )
    }
    return Self(
      directoryURL: storageRoot.appendingPathComponent(
        "desktop-relay",
        isDirectory: true
      )
    )
  }

  func load() throws -> StoredRelayConfiguration {
    do {
      let data = try Data(contentsOf: configurationURL)
      return try JSONDecoder().decode(StoredRelayConfiguration.self, from: data)
    } catch let error as RelayFailure {
      throw error
    } catch {
      if (error as NSError).code == NSFileReadNoSuchFileError {
        throw RelayFailure.invalidSetup(
          "Desktop Relay is not paired. Create a desktop pairing in Xagent "
            + "Settings and launch once with --setup-file."
        )
      }
      throw RelayFailure.invalidSetup(
        "Could not read \(configurationURL.path): \(error.localizedDescription)"
      )
    }
  }

  func save(websocketURL: String) throws {
    let configuration = StoredRelayConfiguration(websocketURL: websocketURL)
    let data = try JSONEncoder().encode(configuration)
    do {
      try FileManager.default.createDirectory(
        at: directoryURL,
        withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700]
      )
      try FileManager.default.setAttributes(
        [.posixPermissions: 0o700],
        ofItemAtPath: directoryURL.path
      )
      try data.write(to: configurationURL, options: .atomic)
      try FileManager.default.setAttributes(
        [.posixPermissions: 0o600],
        ofItemAtPath: configurationURL.path
      )
    } catch {
      throw RelayFailure.invalidSetup(
        "Could not save \(configurationURL.path): \(error.localizedDescription)"
      )
    }
  }

  func removeManagedPairingFileIfNeeded(_ setupFileURL: URL?) {
    guard
      let setupFileURL,
      setupFileURL.standardizedFileURL.deletingLastPathComponent()
        == directoryURL.standardizedFileURL,
      setupFileURL.lastPathComponent == "pairing.json"
    else {
      return
    }
    try? FileManager.default.removeItem(at: setupFileURL)
  }
}
