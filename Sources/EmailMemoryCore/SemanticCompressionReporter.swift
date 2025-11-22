import Foundation

/// Generates human-readable explanations of how semantic compression reduces payload size.
public struct SemanticCompressionReporter: Sendable {
    public init() {}

    /// Builds a report comparing raw omnichannel data to compressed summaries.
    public func makeReport(rawInteractions: [Interaction], chunks: [MemoryChunk]) -> String {
        guard !rawInteractions.isEmpty, !chunks.isEmpty else {
            return "Semantic compression report unavailable: missing interactions or chunks."
        }
        let totalRawCharacters = rawInteractions.reduce(0) { $0 + $1.body.count }
        let levelBuckets = Dictionary(grouping: chunks, by: { $0.level })
        let highestLevel = chunks.map { $0.level }.max() ?? 0
        var lines: [String] = []
        lines.append("Semantic compression overview:")
        lines.append("- Raw interaction characters: \(totalRawCharacters)")
        lines.append("- Stored memory chunks: \(chunks.count)")
        let formatter = NumberFormatter()
        formatter.numberStyle = .percent
        formatter.maximumFractionDigits = 1
        for level in levelBuckets.keys.sorted() {
            let entries = levelBuckets[level] ?? []
            let combinedChars = entries.reduce(0) { $0 + $1.summaryText.count }
            let ratio = totalRawCharacters == 0 ? 0 : Double(combinedChars) / Double(totalRawCharacters)
            let ratioString = formatter.string(from: NSNumber(value: ratio)) ?? "0%"
            lines.append("  • Level \(level): \(entries.count) chunks, \(combinedChars) chars (~\(ratioString) of raw)")
        }
        if let topChunk = levelBuckets[highestLevel]?.first {
            let ratio = totalRawCharacters == 0 ? 0 : Double(topChunk.summaryText.count) / Double(totalRawCharacters)
            let ratioString = formatter.string(from: NSNumber(value: ratio)) ?? "0%"
            lines.append("- Highest level chunk compresses entire thread to ~\(ratioString) of original text.")
        }
        lines.append("- Benefit: embed + summarize once, then reuse vectors for real-time queries instead of streaming gigabytes to the LLM.")
        return lines.joined(separator: "\n")
    }
}
