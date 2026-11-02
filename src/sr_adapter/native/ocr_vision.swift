// On-device OCR only. The Python host supplies a normalized PNG and explicit languages.
import Foundation
import Vision
import ImageIO

do {
    guard CommandLine.arguments.count == 3 else { throw NSError(domain: "arguments", code: 1) }
    let bytes = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
    guard let source = CGImageSourceCreateWithData(bytes as CFData, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "image", code: 1)
    }
    let languages = CommandLine.arguments[2].split(separator: ",").map(String.init)
    let request = VNRecognizeTextRequest()
    request.revision = VNRecognizeTextRequestRevision3
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.recognitionLanguages = languages
    let supported = try request.supportedRecognitionLanguages()
    guard languages.allSatisfy({ supported.contains($0) }) else { throw NSError(domain: "language", code: 1) }
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    var lines: [[String: Any]] = []
    for observation in request.results ?? [] {
        guard let candidate = observation.topCandidates(1).first else { continue }
        let box = observation.boundingBox
        let width = Double(image.width), height = Double(image.height)
        lines.append([
            "text": candidate.string,
            "confidence": Double(candidate.confidence),
            "box": [max(0, Int(floor(box.minX * width))), max(0, Int(floor((1-box.maxY) * height))),
                    min(image.width, Int(ceil(box.maxX * width))), min(image.height, Int(ceil((1-box.minY) * height)))],
            "words": []
        ])
    }
    let output = try JSONSerialization.data(withJSONObject: ["lines": lines], options: [.sortedKeys])
    FileHandle.standardOutput.write(output)
} catch {
    // Source text, paths and platform diagnostics stay out of public errors.
    exit(1)
}
