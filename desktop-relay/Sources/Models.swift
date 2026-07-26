import Foundation

let relayProtocolVersion = 1

enum JSONValue: Codable, Sendable, Equatable {
  case string(String)
  case number(Double)
  case bool(Bool)
  case object([String: JSONValue])
  case array([JSONValue])
  case null

  init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    if container.decodeNil() {
      self = .null
    } else if let value = try? container.decode(Bool.self) {
      self = .bool(value)
    } else if let value = try? container.decode(Double.self) {
      self = .number(value)
    } else if let value = try? container.decode(String.self) {
      self = .string(value)
    } else if let value = try? container.decode([String: JSONValue].self) {
      self = .object(value)
    } else {
      self = .array(try container.decode([JSONValue].self))
    }
  }

  func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .string(let value): try container.encode(value)
    case .number(let value): try container.encode(value)
    case .bool(let value): try container.encode(value)
    case .object(let value): try container.encode(value)
    case .array(let value): try container.encode(value)
    case .null: try container.encodeNil()
    }
  }

  var objectValue: [String: JSONValue]? {
    if case .object(let value) = self { return value }
    return nil
  }

  var stringValue: String? {
    if case .string(let value) = self { return value }
    return nil
  }

  var numberValue: Double? {
    if case .number(let value) = self { return value }
    return nil
  }

  var boolValue: Bool? {
    if case .bool(let value) = self { return value }
    return nil
  }
}

struct PairingSetup: Codable, Sendable {
  let websocketURL: String
  let pairingToken: String?

  enum CodingKeys: String, CodingKey {
    case websocketURL = "websocket_url"
    case pairingToken = "pairing_token"
  }
}

struct RelayCommand: Codable, Sendable {
  let type: String
  let protocolVersion: Int
  let requestID: String
  let command: String
  let payload: [String: JSONValue]

  enum CodingKeys: String, CodingKey {
    case type
    case protocolVersion = "protocol_version"
    case requestID = "request_id"
    case command
    case payload
  }
}

typealias RelayMediaChunkSender =
  @Sendable (_ transferID: String, _ chunkIndex: Int, _ data: Data) async throws -> Void

struct RelayMediaChunk: Codable, Sendable {
  let type = "media_chunk"
  let protocolVersion = relayProtocolVersion
  let requestID: String
  let transferID: String
  let chunkIndex: Int
  let dataBase64: String

  enum CodingKeys: String, CodingKey {
    case type
    case protocolVersion = "protocol_version"
    case requestID = "request_id"
    case transferID = "transfer_id"
    case chunkIndex = "chunk_index"
    case dataBase64 = "data_base64"
  }
}

struct RelayReady: Codable, Sendable {
  let type: String
  let protocolVersion: Int
  let paired: Bool
  let sessionToken: String?

  enum CodingKeys: String, CodingKey {
    case type
    case protocolVersion = "protocol_version"
    case paired
    case sessionToken = "session_token"
  }
}

struct RelayErrorMessage: Codable, Sendable {
  let type: String
  let error: String
}

struct DesktopWindowStatus: Codable, Sendable {
  let type = "status"
  let protocolVersion = relayProtocolVersion
  let attached: Bool
  let windowID: UInt32?
  let displayID: UInt32?
  let targetScope: String?
  let title: String?
  let application: String?
  let bounds: WindowBounds?
  let permissions: [String: Bool]
  let paused: Bool
  let emergencyStopped: Bool

  enum CodingKeys: String, CodingKey {
    case type
    case protocolVersion = "protocol_version"
    case attached
    case windowID = "window_id"
    case displayID = "display_id"
    case targetScope = "target_scope"
    case title
    case application
    case bounds
    case permissions
    case paused
    case emergencyStopped = "emergency_stopped"
  }
}

struct WindowBounds: Codable, Sendable {
  let x: Double
  let y: Double
  let width: Double
  let height: Double
}

struct ViewportPayload: Codable, Sendable {
  let width: Int
  let height: Int
  let devicePixelRatio: Double

  enum CodingKeys: String, CodingKey {
    case width
    case height
    case devicePixelRatio = "device_pixel_ratio"
  }
}

struct NormalizedBounds: Codable, Sendable {
  let x: Double
  let y: Double
  let width: Double
  let height: Double
}

struct ElementPayload: Codable, Sendable {
  let elementID: String
  let bounds: NormalizedBounds
  let label: String?
  let role: String?
  let text: String?
  let metadata: [String: JSONValue]

  enum CodingKeys: String, CodingKey {
    case elementID = "element_id"
    case bounds
    case label
    case role
    case text
    case metadata
  }
}

struct ObservationPayload: Codable, Sendable {
  let screenshotBase64: String
  let viewport: ViewportPayload
  let elements: [ElementPayload]
  let elementsTruncated: Bool
  let elementExtractionFailed: Bool
  let elementExtractionIncomplete: Bool
  let windowID: UInt32?
  let displayID: UInt32?
  let targetScope: String
  let title: String?
  let application: String?
  let paused: Bool
  let emergencyStopped: Bool

  enum CodingKeys: String, CodingKey {
    case screenshotBase64 = "screenshot_base64"
    case viewport
    case elements
    case elementsTruncated = "elements_truncated"
    case elementExtractionFailed = "element_extraction_failed"
    case elementExtractionIncomplete = "element_extraction_incomplete"
    case windowID = "window_id"
    case displayID = "display_id"
    case targetScope = "target_scope"
    case title
    case application
    case paused
    case emergencyStopped = "emergency_stopped"
  }
}

struct ComputerPoint: Sendable {
  let x: Double
  let y: Double

  init(json: [String: JSONValue]) throws {
    guard
      let x = json["x"]?.numberValue,
      let y = json["y"]?.numberValue,
      (0...1).contains(x),
      (0...1).contains(y)
    else {
      throw RelayFailure.invalidAction("target coordinates must be between 0 and 1")
    }
    self.x = x
    self.y = y
  }
}

enum RelayFailure: LocalizedError, Sendable {
  case invalidSetup(String)
  case invalidMessage(String)
  case invalidAction(String)
  case permission(String)
  case windowUnavailable(String)
  case paused
  case emergencyStopped

  var errorDescription: String? {
    switch self {
    case .invalidSetup(let message),
      .invalidMessage(let message),
      .invalidAction(let message),
      .permission(let message),
      .windowUnavailable(let message):
      message
    case .paused:
      "Desktop Relay is paused by the user."
    case .emergencyStopped:
      "Desktop Relay emergency stop is active. Re-authorize a window or display to continue."
    }
  }
}

func jsonValue<T: Encodable>(_ value: T) throws -> JSONValue {
  let data = try JSONEncoder().encode(value)
  return try JSONDecoder().decode(JSONValue.self, from: data)
}
