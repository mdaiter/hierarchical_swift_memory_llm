import Foundation

/// Builds hierarchical memory chunks from omnichannel interactions.
public struct OmnichannelMemoryBuilder: Sendable {
    public let openAI: OpenAIClient

    public init(openAI: OpenAIClient) {
        self.openAI = openAI
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
        var levelZero: [MemoryChunk] = []
        for group in chunksOfInteractions {
            let summary = try await openAI.summarizeInteractions(group)
            let embedding = try await openAI.embedText(summary)
            let participants = participantsFor(group)
            let sourceKinds = Array(Set(group.map { $0.sourceKind }))
            let chunk = MemoryChunk(
                id: UUID().uuidString,
                level: 0,
                summaryText: summary,
                embedding: embedding,
                sourceInteractionIds: group.map { $0.id },
                participants: participants,
                sourceKinds: sourceKinds
            )
            levelZero.append(chunk)
            allChunks.append(chunk)
        }

        var currentLevel = levelZero
        var level = 1
        while currentLevel.count > 1 {
            let grouped = chunk(currentLevel, size: normalizedGroupSize)
            var nextLevel: [MemoryChunk] = []
            for aggregate in grouped where !aggregate.isEmpty {
                let combined = aggregate
                    .map { "Chunk \($0.id) (level \($0.level)):\n\($0.summaryText)" }
                    .joined(separator: "\n\n---\n\n")
                let summary = try await openAI.summarizeTextBlock(
                    combined,
                    taskDescription: "Summarize combined semantic memories, keeping decisions, concerns, and unresolved items."
                )
                let embedding = try await openAI.embedText(summary)
                let participants = Array(Set(aggregate.flatMap { $0.participants })).sorted()
                let kinds = Array(Set(aggregate.flatMap { $0.sourceKinds }))
                let chunk = MemoryChunk(
                    id: UUID().uuidString,
                    level: level,
                    summaryText: summary,
                    embedding: embedding,
                    sourceInteractionIds: aggregate.flatMap { $0.sourceInteractionIds },
                    participants: participants,
                    sourceKinds: kinds
                )
                nextLevel.append(chunk)
                allChunks.append(chunk)
            }
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
}
