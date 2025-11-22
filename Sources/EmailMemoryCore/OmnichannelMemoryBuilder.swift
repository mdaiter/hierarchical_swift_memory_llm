import Foundation
import CryptoKit

/// Builds hierarchical memory chunks from omnichannel interactions.
public struct OmnichannelMemoryBuilder: Sendable {
    public let openAI: OpenAIClient
    private let requestLimiter: AsyncSemaphore
    private let chunkCache: ChunkCache?

    public init(
        openAI: OpenAIClient,
        maxConcurrentRequests: Int = 4,
        chunkCache: ChunkCache? = nil
    ) {
        self.openAI = openAI
        self.requestLimiter = AsyncSemaphore(value: max(1, maxConcurrentRequests))
        self.chunkCache = chunkCache
    }

    /// Builds semantic memory chunks across any supported channel.
    public func buildMemory(
        interactions: [Interaction],
        chunkSize: Int = 12,
        groupSize: Int = 5
    ) async throws -> [MemoryChunk] {
        guard !interactions.isEmpty else { return [] }
        let sorted = interactions.sorted { $0.timestamp < $1.timestamp }
        let normalizedChunkSize = max(1, chunkSize)
        let normalizedGroupSize = max(2, groupSize)
        var allChunks: [MemoryChunk] = []

        let chunksOfInteractions = chunk(sorted, size: normalizedChunkSize)
        let levelZero = try await buildLevelZeroChunks(groups: chunksOfInteractions)
        allChunks.append(contentsOf: levelZero)

        var currentLevel = levelZero
        var level = 1
        while currentLevel.count > 1 {
            let grouped = chunk(currentLevel, size: normalizedGroupSize)
            let nextLevel = try await buildAggregatedChunks(groups: grouped, level: level)
            allChunks.append(contentsOf: nextLevel)
            guard !nextLevel.isEmpty, nextLevel.count < currentLevel.count else { break }
            currentLevel = nextLevel
            level += 1
        }

        return allChunks
    }

    private func chunk<T>(_ array: [T], size: Int) -> [[T]] {
        guard size > 0 else { return [array] }
        var result: [[T]] = []
        var index = 0
        while index < array.count {
            let end = min(index + size, array.count)
            result.append(Array(array[index..<end]))
            index = end
        }
        return result
    }

    private func participantsFor(_ interactions: [Interaction]) -> [String] {
        Array(
            Set(
                interactions.flatMap { interaction in
                    [interaction.from] + interaction.to
                }
            )
        ).sorted { $0.lowercased() < $1.lowercased() }
    }

    private func buildLevelZeroChunks(groups: [[Interaction]]) async throws -> [MemoryChunk] {
        guard !groups.isEmpty else { return [] }
        return try await withThrowingTaskGroup(of: (Int, MemoryChunk).self) { group in
            for (idx, interactions) in groups.enumerated() {
                group.addTask {
                    let key = self.cacheKey(forInteractions: interactions)
                    if let cached = await self.chunkCache?.chunk(forKey: key) {
                        return (idx, cached)
                    }
                    await self.requestLimiter.wait()
                    defer { Task { await self.requestLimiter.signal() } }
                    let summary = try await self.openAI.summarizeInteractions(interactions)
                    let embedding = try await self.openAI.embedText(summary)
                    let participants = self.participantsFor(interactions)
                    let sourceKinds = Array(Set(interactions.map { $0.sourceKind }))
                    let latestTimestamp = interactions.map { $0.timestamp }.max()
                    let chunk = MemoryChunk(
                        id: UUID().uuidString,
                        level: 0,
                        summaryText: summary,
                        embedding: embedding,
                        sourceInteractionIds: interactions.map { $0.id },
                        participants: participants,
                        sourceKinds: sourceKinds,
                        latestTimestamp: latestTimestamp
                    )
                    await self.chunkCache?.store(chunk, forKey: key)
                    return (idx, chunk)
                }
            }

            var results: [(Int, MemoryChunk)] = []
            for try await chunk in group {
                results.append(chunk)
            }
            return results.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
    }

    private func buildAggregatedChunks(groups: [[MemoryChunk]], level: Int) async throws -> [MemoryChunk] {
        guard !groups.isEmpty else { return [] }
        return try await withThrowingTaskGroup(of: MemoryChunk?.self) { group in
            for aggregate in groups {
                group.addTask {
                    guard !aggregate.isEmpty else { return nil }
                    let key = self.cacheKey(forAggregates: aggregate, level: level)
                    if let cached = await self.chunkCache?.chunk(forKey: key) {
                        return cached
                    }
                    await self.requestLimiter.wait()
                    defer { Task { await self.requestLimiter.signal() } }
                    let combined = aggregate
                        .map { "Chunk \($0.id) (level \($0.level)):\n\($0.summaryText)" }
                        .joined(separator: "\n\n---\n\n")
                    let summary = try await self.openAI.summarizeTextBlock(
                        combined,
                        taskDescription: "Summarize combined semantic memories, keeping decisions, concerns, and unresolved items."
                    )
                    let embedding = try await self.openAI.embedText(summary)
                    let participants = Array(Set(aggregate.flatMap { $0.participants })).sorted()
                    let kinds = Array(Set(aggregate.flatMap { $0.sourceKinds }))
                    let latestTimestamp = aggregate.compactMap { $0.latestTimestamp }.max()
                    let chunk = MemoryChunk(
                        id: UUID().uuidString,
                        level: level,
                        summaryText: summary,
                        embedding: embedding,
                        sourceInteractionIds: aggregate.flatMap { $0.sourceInteractionIds },
                        participants: participants,
                        sourceKinds: kinds,
                        latestTimestamp: latestTimestamp
                    )
                    await self.chunkCache?.store(chunk, forKey: key)
                    return chunk
                }
            }

            var combined: [MemoryChunk] = []
            for try await chunk in group {
                if let chunk {
                    combined.append(chunk)
                }
            }
            return combined
        }
    }

    private func cacheKey(forInteractions interactions: [Interaction]) -> String {
        let descriptor = interactions
            .map { "\($0.id):\($0.timestamp.timeIntervalSince1970)" }
            .joined(separator: "|")
        return "L0-\(Self.hash(of: descriptor))"
    }

    private func cacheKey(forAggregates aggregates: [MemoryChunk], level: Int) -> String {
        let descriptor = aggregates
            .flatMap { $0.sourceInteractionIds }
            .sorted()
            .joined(separator: "|")
        return "L\(level)-\(Self.hash(of: descriptor))"
    }

    private static func hash(of string: String) -> String {
        let digest = SHA256.hash(data: Data(string.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
