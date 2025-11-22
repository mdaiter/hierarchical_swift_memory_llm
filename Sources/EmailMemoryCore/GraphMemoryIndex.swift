import Foundation

/// A graph-based approximate nearest neighbor index over MemoryChunk embeddings.
public final class GraphMemoryIndex: @unchecked Sendable {

    // MARK: - Nested Types

    private struct Node: Identifiable {
        let chunk: MemoryChunk
        let participants: Set<String>
        let sourceKinds: Set<SourceKind>
        var neighbors: [Int]

        var id: String { chunk.id }
        var embedding: [Double] { chunk.embedding }
        var latestTimestamp: Date? { chunk.latestTimestamp }
    }

    // MARK: - Stored Properties

    private var nodes: [Node] = []
    private var idToIndex: [String: Int] = [:]
    private let neighborCount: Int
    private let recencyHalfLifeDays: Double

    // MARK: - Init

    public init(
        chunks: [MemoryChunk],
        neighborCount: Int = 8,
        recencyHalfLifeDays: Double = 30.0
    ) {
        self.neighborCount = max(1, neighborCount)
        self.recencyHalfLifeDays = recencyHalfLifeDays
        buildGraph(from: chunks)
    }

    // MARK: - Public

    public var count: Int {
        nodes.count
    }

    public func isEmpty() -> Bool {
        nodes.isEmpty
    }

    /// Approximate nearest neighbor search using greedy graph traversal and metadata filters.
    public func search(
        queryEmbedding: [Double],
        k: Int,
        filter: SearchFilter? = nil,
        includeRecency: Bool = true
    ) -> [MemoryChunk] {
        guard !nodes.isEmpty, !queryEmbedding.isEmpty else {
            return []
        }

        let k = max(1, k)
        let targetCandidates = min(nodes.count, max(k * 4, k + 2))
        let entryIndices = initialEntryPoints(queryEmbedding: queryEmbedding, maxEntries: min(8, nodes.count))

        var visited = Set<Int>()
        var frontier = PriorityQueue<(index: Int, score: Double)>(sort: { $0.score > $1.score })

        for idx in entryIndices {
            guard idx >= 0 && idx < nodes.count else { continue }
            let node = nodes[idx]
            let sim = cosineSimilarity(node.embedding, queryEmbedding)
            let score = combinedScore(similarity: sim, node: node, includeRecency: includeRecency)
            frontier.enqueue((index: idx, score: score))
        }

        var candidates: [(index: Int, score: Double)] = []
        let maxExplorations = min(nodes.count, targetCandidates * 5)

        while let current = frontier.dequeue(), visited.count < maxExplorations {
            if visited.contains(current.index) {
                continue
            }
            visited.insert(current.index)

            let node = nodes[current.index]
            if filter?.isChunkEligible(node.chunk) ?? true {
                candidates.append(current)
                if candidates.count >= targetCandidates {
                    break
                }
            }

            for neighborIdx in node.neighbors {
                if visited.contains(neighborIdx) {
                    continue
                }
                let neighbor = nodes[neighborIdx]
                let sim = cosineSimilarity(neighbor.embedding, queryEmbedding)
                let score = combinedScore(similarity: sim, node: neighbor, includeRecency: includeRecency)
                frontier.enqueue((index: neighborIdx, score: score))
            }
        }

        if candidates.isEmpty {
            return []
        }

        let mmr = mmrSelectCandidates(
            candidates: candidates,
            queryEmbedding: queryEmbedding,
            k: k
        )

        return mmr.compactMap { candidate in
            guard candidate.index >= 0 && candidate.index < nodes.count else { return nil }
            return nodes[candidate.index].chunk
        }
    }

    // MARK: - Graph Construction

    private func buildGraph(from chunks: [MemoryChunk]) {
        nodes.removeAll(keepingCapacity: true)
        idToIndex.removeAll(keepingCapacity: true)

        for (idx, chunk) in chunks.enumerated() {
            let node = Node(
                chunk: chunk,
                participants: Set(chunk.participants.map { $0.lowercased() }),
                sourceKinds: Set(chunk.sourceKinds),
                neighbors: []
            )
            nodes.append(node)
            idToIndex[chunk.id] = idx
        }

        guard nodes.count > 1 else { return }

        let n = nodes.count
        var similarityMatrix: [[Double]] = Array(
            repeating: Array(repeating: 0.0, count: n),
            count: n
        )

        for i in 0..<n {
            for j in (i + 1)..<n {
                let sim = cosineSimilarity(nodes[i].embedding, nodes[j].embedding)
                similarityMatrix[i][j] = sim
                similarityMatrix[j][i] = sim
            }
        }

        for i in 0..<n {
            let sims = similarityMatrix[i]
            var indexed: [(idx: Int, sim: Double)] = sims.enumerated()
                .filter { $0.offset != i }
                .map { (idx: $0.offset, sim: $0.element) }
            indexed.sort { $0.sim > $1.sim }
            nodes[i].neighbors = Array(indexed.prefix(neighborCount).map { $0.idx })
        }

        // Ensure symmetry
        for i in 0..<n {
            for neighborIdx in nodes[i].neighbors {
                if !nodes[neighborIdx].neighbors.contains(i) {
                    nodes[neighborIdx].neighbors.append(i)
                }
            }
        }
    }

    // MARK: - Helpers

    private func initialEntryPoints(queryEmbedding: [Double], maxEntries: Int) -> [Int] {
        if nodes.isEmpty { return [] }
        var scored: [(index: Int, score: Double)] = []
        for idx in 0..<nodes.count {
            let node = nodes[idx]
            let sim = cosineSimilarity(node.embedding, queryEmbedding)
            scored.append((idx, sim))
        }
        scored.sort { $0.score > $1.score }
        return Array(scored.prefix(maxEntries).map { $0.index })
    }

    private func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        guard !a.isEmpty, a.count == b.count else { return 0.0 }
        var dot: Double = 0
        var normA: Double = 0
        var normB: Double = 0
        for i in 0..<a.count {
            let va = a[i]
            let vb = b[i]
            dot += va * vb
            normA += va * va
            normB += vb * vb
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom == 0 ? 0 : dot / denom
    }

    private func combinedScore(
        similarity: Double,
        node: Node,
        includeRecency: Bool
    ) -> Double {
        guard includeRecency, let ts = node.latestTimestamp else {
            return similarity
        }
        let now = Date()
        let seconds = now.timeIntervalSince(ts)
        let days = seconds / (60 * 60 * 24)
        let halfLife = max(recencyHalfLifeDays, 1.0)
        let decay = exp(-log(2.0) * days / halfLife)
        let alpha = 0.8
        return alpha * similarity + (1 - alpha) * decay
    }

    private func mmrSelectCandidates(
        candidates: [(index: Int, score: Double)],
        queryEmbedding: [Double],
        k: Int,
        lambda: Double = 0.7
    ) -> [(index: Int, score: Double)] {
        guard !candidates.isEmpty else { return [] }
        var selected: [(index: Int, score: Double)] = []
        var remaining = candidates

        func embedding(for index: Int) -> [Double] {
            nodes[index].embedding
        }

        func cosine(_ lhs: Int, _ rhs: Int) -> Double {
            cosineSimilarity(embedding(for: lhs), embedding(for: rhs))
        }

        while selected.count < k && !remaining.isEmpty {
            var bestIdx = 0
            var bestScore = -Double.infinity
            for (offset, candidate) in remaining.enumerated() {
                let nodeIndex = candidate.index
                let simToQuery = cosineSimilarity(embedding(for: nodeIndex), queryEmbedding)
                var maxSimToSelected = 0.0
                if !selected.isEmpty {
                    maxSimToSelected = selected
                        .map { cosine(nodeIndex, $0.index) }
                        .max() ?? 0
                }
                let mmrScore = lambda * simToQuery - (1 - lambda) * maxSimToSelected
                if mmrScore > bestScore {
                    bestScore = mmrScore
                    bestIdx = offset
                }
            }
            let chosen = remaining.remove(at: bestIdx)
            selected.append(chosen)
        }

        return selected
    }
}
