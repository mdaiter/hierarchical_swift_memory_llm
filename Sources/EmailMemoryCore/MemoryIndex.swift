import Foundation

/// In-memory cosine similarity index for memory chunks.
public final class MemoryIndex: @unchecked Sendable {
    private var chunks: [MemoryChunk]

    public init(chunks: [MemoryChunk]) {
        self.chunks = chunks
    }

    public func add(_ chunk: MemoryChunk) {
        chunks.append(chunk)
    }

    /// Returns chunks sorted by cosine similarity.
    public func search(
        queryEmbedding: [Double],
        k: Int,
        maxLevel: Int? = nil
    ) -> [MemoryChunk] {
        searchWithScores(queryEmbedding: queryEmbedding, maxResults: k, maxLevel: maxLevel).map { $0.chunk }
    }

    /// Returns chunks with similarity scores for downstream scoring.
    public func searchWithScores(
        queryEmbedding: [Double],
        maxResults: Int,
        maxLevel: Int? = nil
    ) -> [(chunk: MemoryChunk, similarity: Double)] {
        guard !chunks.isEmpty, maxResults > 0 else { return [] }
        let filtered: [MemoryChunk]
        if let maxLevel {
            filtered = chunks.filter { $0.level <= maxLevel }
        } else {
            filtered = chunks
        }
        guard !filtered.isEmpty else { return [] }
        let scored = filtered.map { chunk in
            (chunk, cosineSimilarity(queryEmbedding, chunk.embedding))
        }
        return scored.sorted { lhs, rhs in
            if lhs.1 == rhs.1 {
                return lhs.0.level < rhs.0.level
            }
            return lhs.1 > rhs.1
        }
        .prefix(min(maxResults, scored.count))
        .map { ($0.0, $0.1) }
    }

    private func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        guard !a.isEmpty, !b.isEmpty else { return 0 }
        let count = min(a.count, b.count)
        guard count > 0 else { return 0 }
        var dot: Double = 0
        var magA: Double = 0
        var magB: Double = 0
        for idx in 0..<count {
            let x = a[idx]
            let y = b[idx]
            dot += x * y
            magA += x * x
            magB += y * y
        }
        guard magA > 0 && magB > 0 else { return 0 }
        return dot / (sqrt(magA) * sqrt(magB))
    }
}
