import Foundation

private struct RelayHello: Codable {
  let type = "hello"
  let protocolVersion = relayProtocolVersion
  let clientID: String
  let clientName: String
  let pairingToken: String?
  let sessionToken: String?

  enum CodingKeys: String, CodingKey {
    case type
    case protocolVersion = "protocol_version"
    case clientID = "client_id"
    case clientName = "client_name"
    case pairingToken = "pairing_token"
    case sessionToken = "session_token"
  }
}

private struct RelayPing: Codable {
  let type = "ping"
  let protocolVersion = relayProtocolVersion

  enum CodingKeys: String, CodingKey {
    case type
    case protocolVersion = "protocol_version"
  }
}

private struct RelayResponse: Codable {
  let type = "response"
  let protocolVersion = relayProtocolVersion
  let requestID: String
  let success: Bool
  let result: [String: JSONValue]?
  let error: String

  enum CodingKeys: String, CodingKey {
    case type
    case protocolVersion = "protocol_version"
    case requestID = "request_id"
    case success
    case result
    case error
  }
}

private struct MessageKind: Codable {
  let type: String
}

private final class RelaySessionDelegate: NSObject, URLSessionTaskDelegate,
  @unchecked Sendable
{
  func urlSession(
    _ session: URLSession,
    task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest,
    completionHandler: @escaping @Sendable (URLRequest?) -> Void
  ) {
    completionHandler(nil)
  }
}

actor RelayClient {
  typealias CommandHandler = @Sendable (RelayCommand) async throws -> [String: JSONValue]
  typealias StatusProvider = @Sendable () async -> DesktopWindowStatus

  private let setup: PairingSetup
  private let commandHandler: CommandHandler
  private let statusProvider: StatusProvider
  private let clientID: String
  private let clientName = Host.current().localizedName ?? "Mac"
  private let credentialAccount: String
  private let urlSession: URLSession
  private var socket: URLSessionWebSocketTask?
  private var shouldRun = true
  private var paired = false

  init(
    setup: PairingSetup,
    commandHandler: @escaping CommandHandler,
    statusProvider: @escaping StatusProvider
  ) throws {
    guard
      let components = URLComponents(string: setup.websocketURL),
      ["ws", "wss"].contains(components.scheme?.lowercased() ?? ""),
      components.host != nil,
      components.user == nil,
      components.password == nil,
      components.query == nil,
      components.fragment == nil
    else {
      throw RelayFailure.invalidSetup(
        "websocket_url must be an absolute ws:// or wss:// URL without credentials or query parameters"
      )
    }
    guard !setup.pairingToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw RelayFailure.invalidSetup("pairing_token must not be empty")
    }
    self.setup = setup
    self.commandHandler = commandHandler
    self.statusProvider = statusProvider
    let port = components.port.map(String.init) ?? ""
    credentialAccount = [
      components.scheme ?? "",
      components.host ?? "",
      port,
      components.path,
    ].joined(separator: ":")
    urlSession = URLSession(
      configuration: .ephemeral,
      delegate: RelaySessionDelegate(),
      delegateQueue: nil
    )
    let defaults = UserDefaults.standard
    if let stored = defaults.string(forKey: "client-id"), !stored.isEmpty {
      clientID = stored
    } else {
      clientID = UUID().uuidString.lowercased()
      defaults.set(clientID, forKey: "client-id")
    }
  }

  func runForever() async {
    var attempt = 0
    while shouldRun {
      do {
        try await runConnection()
        attempt = 0
      } catch is CancellationError {
        return
      } catch {
        attempt += 1
        let seconds = min(30, max(1, 1 << min(attempt - 1, 5)))
        FileHandle.standardError.write(
          Data(
            "Desktop relay disconnected: \(error.localizedDescription). Retrying in \(seconds)s.\n"
              .utf8)
        )
        try? await Task.sleep(for: .seconds(seconds))
      }
    }
  }

  func sendCurrentStatus() async {
    guard socket != nil else { return }
    do {
      try await send(await statusProvider())
    } catch {
      FileHandle.standardError.write(
        Data("Could not send desktop relay status: \(error.localizedDescription)\n".utf8)
      )
    }
  }

  func stop() {
    shouldRun = false
    socket?.cancel(with: .normalClosure, reason: nil)
    socket = nil
    urlSession.invalidateAndCancel()
  }

  private func runConnection() async throws {
    guard let url = URL(string: setup.websocketURL) else {
      throw RelayFailure.invalidSetup("websocket_url is invalid")
    }
    let task = urlSession.webSocketTask(with: url)
    socket = task
    paired = false
    task.resume()
    defer {
      task.cancel(with: .goingAway, reason: nil)
      if socket === task { socket = nil }
    }

    let savedSession = KeychainStore.loadSessionToken(account: credentialAccount)
    let hello = RelayHello(
      clientID: clientID,
      clientName: clientName,
      pairingToken: savedSession == nil ? setup.pairingToken : nil,
      sessionToken: savedSession
    )
    try await send(hello)

    async let keepalive: Void = pingLoop(task)
    do {
      while shouldRun {
        let message = try await task.receive()
        let data: Data
        switch message {
        case .string(let text): data = Data(text.utf8)
        case .data(let bytes): data = bytes
        @unknown default:
          throw RelayFailure.invalidMessage("unsupported WebSocket message")
        }
        try await handleMessage(data)
      }
    } catch {
      task.cancel(with: .goingAway, reason: nil)
      _ = await keepalive
      throw error
    }
    _ = await keepalive
  }

  private func handleMessage(_ data: Data) async throws {
    let kind = try JSONDecoder().decode(MessageKind.self, from: data)
    switch kind.type {
    case "ready":
      let ready = try JSONDecoder().decode(RelayReady.self, from: data)
      guard ready.protocolVersion == relayProtocolVersion else {
        throw RelayFailure.invalidMessage("desktop relay protocol version mismatch")
      }
      if let token = ready.sessionToken {
        try KeychainStore.saveSessionToken(
          token,
          account: credentialAccount
        )
      }
      paired = true
      try await send(await statusProvider())
    case "command":
      guard paired else {
        throw RelayFailure.invalidMessage("command arrived before relay was ready")
      }
      let command = try JSONDecoder().decode(RelayCommand.self, from: data)
      await handleCommand(command)
    case "pong":
      return
    case "error":
      let message = try JSONDecoder().decode(RelayErrorMessage.self, from: data)
      if message.error.localizedCaseInsensitiveContains("invalid or expired") {
        KeychainStore.deleteSessionToken(account: credentialAccount)
      }
      throw RelayFailure.invalidMessage(message.error)
    default:
      throw RelayFailure.invalidMessage("unsupported relay message type \(kind.type)")
    }
  }

  private func handleCommand(_ command: RelayCommand) async {
    guard command.protocolVersion == relayProtocolVersion else {
      await sendFailure(command, "desktop relay protocol version mismatch")
      return
    }
    do {
      let result = try await commandHandler(command)
      try await send(
        RelayResponse(
          requestID: command.requestID,
          success: true,
          result: result,
          error: ""
        )
      )
    } catch {
      await sendFailure(command, error.localizedDescription)
    }
  }

  private func sendFailure(_ command: RelayCommand, _ error: String) async {
    try? await send(
      RelayResponse(
        requestID: command.requestID,
        success: false,
        result: nil,
        error: String(error.prefix(2_000))
      )
    )
  }

  private func pingLoop(_ task: URLSessionWebSocketTask) async {
    while shouldRun, socket === task {
      try? await Task.sleep(for: .seconds(20))
      guard shouldRun, socket === task else { return }
      try? await send(RelayPing())
    }
  }

  private func send<T: Encodable>(_ value: T) async throws {
    guard let socket else {
      throw RelayFailure.invalidMessage("desktop relay is not connected")
    }
    let data = try JSONEncoder().encode(value)
    guard data.count <= 12 * 1024 * 1024 else {
      throw RelayFailure.invalidMessage("desktop relay message is too large")
    }
    guard let text = String(data: data, encoding: .utf8) else {
      throw RelayFailure.invalidMessage(
        "desktop relay message could not be encoded as UTF-8"
      )
    }
    try await socket.send(.string(text))
  }
}
