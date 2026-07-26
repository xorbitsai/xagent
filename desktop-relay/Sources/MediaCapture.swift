import AVFoundation
import CoreMedia
import CryptoKit
import Foundation
import ScreenCaptureKit

private let maximumRelayMediaBytes = 32 * 1024 * 1024
private let relayMediaChunkBytes = 256 * 1024

enum DesktopMediaKind: String, Sendable {
  case audio
  case video
}

enum DesktopCaptureSource: Sendable {
  case window(CGWindowID)
  case display(CGDirectDisplayID)
}

struct MediaArtifactPayload: Codable, Sendable {
  let transferID: String
  let mimeType: String
  let mediaKind: String
  let durationMilliseconds: Int
  let chunkCount: Int
  let sizeBytes: Int
  let sha256: String

  enum CodingKeys: String, CodingKey {
    case transferID = "transfer_id"
    case mimeType = "mime_type"
    case mediaKind = "media_kind"
    case durationMilliseconds = "duration_ms"
    case chunkCount = "chunk_count"
    case sizeBytes = "size_bytes"
    case sha256
  }
}

final class DesktopMediaRecorder: NSObject, SCStreamOutput, SCStreamDelegate,
  @unchecked Sendable
{
  private let writer: AVAssetWriter
  private let audioInput: AVAssetWriterInput
  private let videoInput: AVAssetWriterInput?
  private let sampleQueue = DispatchQueue(
    label: "ai.xagent.desktop-relay.media",
    qos: .userInitiated
  )
  private var firstSampleTime: CMTime?
  private var recordingError: Error?

  private init(
    outputURL: URL,
    mediaKind: DesktopMediaKind,
    width: Int,
    height: Int
  ) throws {
    writer = try AVAssetWriter(
      outputURL: outputURL,
      fileType: mediaKind == .audio ? .m4a : .mp4
    )
    audioInput = AVAssetWriterInput(
      mediaType: .audio,
      outputSettings: [
        AVFormatIDKey: kAudioFormatMPEG4AAC,
        AVSampleRateKey: 48_000,
        AVNumberOfChannelsKey: 2,
        AVEncoderBitRateKey: 128_000,
      ]
    )
    audioInput.expectsMediaDataInRealTime = true
    guard writer.canAdd(audioInput) else {
      throw RelayFailure.invalidMessage(
        "could not configure the desktop audio encoder"
      )
    }
    writer.add(audioInput)

    if mediaKind == .video {
      let input = AVAssetWriterInput(
        mediaType: .video,
        outputSettings: [
          AVVideoCodecKey: AVVideoCodecType.h264,
          AVVideoWidthKey: width,
          AVVideoHeightKey: height,
          AVVideoCompressionPropertiesKey: [
            AVVideoAverageBitRateKey: 1_500_000,
            AVVideoExpectedSourceFrameRateKey: 30,
            AVVideoMaxKeyFrameIntervalKey: 60,
          ],
        ]
      )
      input.expectsMediaDataInRealTime = true
      guard writer.canAdd(input) else {
        throw RelayFailure.invalidMessage(
          "could not configure the desktop video encoder"
        )
      }
      writer.add(input)
      videoInput = input
    } else {
      videoInput = nil
    }
    super.init()
  }

  static func capture(
    source: DesktopCaptureSource,
    frame: CGRect,
    mediaKind: DesktopMediaKind,
    durationMilliseconds: Int,
    transferID: String,
    sendMediaChunk: @escaping RelayMediaChunkSender
  ) async throws -> MediaArtifactPayload {
    let duration = max(1_000, min(30_000, durationMilliseconds))
    let dimensions = videoDimensions(frame)
    let suffix = mediaKind == .audio ? "m4a" : "mp4"
    let outputURL = FileManager.default.temporaryDirectory
      .appendingPathComponent(
        "xagent-\(UUID().uuidString).\(suffix)",
        isDirectory: false
      )
    defer { try? FileManager.default.removeItem(at: outputURL) }

    let recorder = try DesktopMediaRecorder(
      outputURL: outputURL,
      mediaKind: mediaKind,
      width: dimensions.width,
      height: dimensions.height
    )
    let content = try await SCShareableContent.excludingDesktopWindows(
      false,
      onScreenWindowsOnly: true
    )
    let contentFilter: SCContentFilter
    switch source {
    case .window(let windowID):
      guard let window = content.windows.first(where: { $0.windowID == windowID }) else {
        throw RelayFailure.windowUnavailable(
          "the authorized window is no longer available for media capture"
        )
      }
      contentFilter = SCContentFilter(desktopIndependentWindow: window)
    case .display(let displayID):
      guard let display = content.displays.first(where: { $0.displayID == displayID })
      else {
        throw RelayFailure.windowUnavailable(
          "the authorized display is no longer available for media capture"
        )
      }
      contentFilter = SCContentFilter(display: display, excludingWindows: [])
    }
    let configuration = SCStreamConfiguration()
    configuration.width = mediaKind == .video ? dimensions.width : 2
    configuration.height = mediaKind == .video ? dimensions.height : 2
    configuration.minimumFrameInterval = CMTime(value: 1, timescale: 30)
    configuration.queueDepth = 5
    configuration.showsCursor = true
    configuration.capturesAudio = true
    configuration.sampleRate = 48_000
    configuration.channelCount = 2
    configuration.excludesCurrentProcessAudio = true
    configuration.ignoreShadowsSingleWindow = true

    let stream = SCStream(
      filter: contentFilter,
      configuration: configuration,
      delegate: recorder
    )
    try stream.addStreamOutput(
      recorder,
      type: .audio,
      sampleHandlerQueue: recorder.sampleQueue
    )
    if mediaKind == .video {
      try stream.addStreamOutput(
        recorder,
        type: .screen,
        sampleHandlerQueue: recorder.sampleQueue
      )
    }
    try await stream.startCapture()
    do {
      try await Task.sleep(for: .milliseconds(duration))
      try await stream.stopCapture()
    } catch {
      try? await stream.stopCapture()
      throw error
    }
    try await recorder.finish()

    let attributes = try FileManager.default.attributesOfItem(atPath: outputURL.path)
    let fileSize = (attributes[.size] as? NSNumber)?.intValue ?? 0
    guard fileSize > 0 else {
      throw RelayFailure.invalidMessage(
        "the authorized desktop target produced no media data"
      )
    }
    guard fileSize <= maximumRelayMediaBytes else {
      throw RelayFailure.invalidMessage(
        "the captured media exceeds 32 MiB; capture a shorter segment"
      )
    }
    let file = try FileHandle(forReadingFrom: outputURL)
    defer { try? file.close() }
    var chunkIndex = 0
    var transferredBytes = 0
    var hasher = SHA256()
    while let data = try file.read(upToCount: relayMediaChunkBytes), !data.isEmpty {
      hasher.update(data: data)
      try await sendMediaChunk(transferID, chunkIndex, data)
      transferredBytes += data.count
      chunkIndex += 1
    }
    guard transferredBytes == fileSize, chunkIndex > 0 else {
      throw RelayFailure.invalidMessage(
        "the desktop media file changed during transfer"
      )
    }
    let checksum = hasher.finalize().map { String(format: "%02x", $0) }.joined()
    return MediaArtifactPayload(
      transferID: transferID,
      mimeType: mediaKind == .audio ? "audio/mp4" : "video/mp4",
      mediaKind: mediaKind.rawValue,
      durationMilliseconds: duration,
      chunkCount: chunkIndex,
      sizeBytes: transferredBytes,
      sha256: checksum
    )
  }

  func stream(
    _ stream: SCStream,
    didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
    of outputType: SCStreamOutputType
  ) {
    guard sampleBuffer.isValid, CMSampleBufferDataIsReady(sampleBuffer) else {
      return
    }
    if firstSampleTime == nil {
      let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
      guard writer.startWriting() else {
        recordingError =
          writer.error
          ?? RelayFailure.invalidMessage("could not start the desktop media encoder")
        return
      }
      writer.startSession(atSourceTime: presentationTime)
      firstSampleTime = presentationTime
    }
    let input = outputType == .audio ? audioInput : videoInput
    if let input, input.isReadyForMoreMediaData, !input.append(sampleBuffer) {
      recordingError =
        writer.error
        ?? RelayFailure.invalidMessage("could not encode desktop media")
    }
  }

  func stream(_ stream: SCStream, didStopWithError error: Error) {
    sampleQueue.async { [weak self] in
      self?.recordingError = error
    }
  }

  private func finish() async throws {
    try await withCheckedThrowingContinuation {
      (continuation: CheckedContinuation<Void, Error>) in
      sampleQueue.async { [self] in
        if let recordingError {
          writer.cancelWriting()
          continuation.resume(throwing: recordingError)
          return
        }
        guard firstSampleTime != nil else {
          writer.cancelWriting()
          continuation.resume(
            throwing: RelayFailure.invalidMessage(
              "the authorized desktop target produced no media samples"
            )
          )
          return
        }
        audioInput.markAsFinished()
        videoInput?.markAsFinished()
        writer.finishWriting {
          if self.writer.status == .completed {
            continuation.resume()
          } else {
            continuation.resume(
              throwing: self.writer.error
                ?? RelayFailure.invalidMessage(
                  "could not finish the desktop media file"
                )
            )
          }
        }
      }
    }
  }

  private static func videoDimensions(_ frame: CGRect) -> (
    width: Int,
    height: Int
  ) {
    let maximumWidth = 1_920.0
    let scale = min(1, maximumWidth / max(1, frame.width))
    let width = max(2, Int(frame.width * scale) / 2 * 2)
    let height = max(2, Int(frame.height * scale) / 2 * 2)
    return (width, height)
  }
}
