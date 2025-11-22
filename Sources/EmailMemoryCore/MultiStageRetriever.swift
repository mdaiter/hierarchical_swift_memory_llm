import Foundation

/// Performs multi-stage retrieval with filtering, vector search, re-ranking, and context assembly.
public struct MultiStageRetriever: Sendable {
    public let openAI: OpenAIClient

    public init(openAI: OpenAIClient) {
        self.openAI = openAI
    }

    /// Retrieves and assembles context across persona/entity/situation/chunks.
    public func retrieve(
        query: String,
        persona: PersonaCard?,
        entityCards: [EntityCard],
        situation: SituationCard?,
        chunks: [MemoryChunk],
        k: Int = 8
    ) async throws -> (selectedChunks: [MemoryChunk], enrichedContext: String) {
        guard !chunks.isEmpty else {
            let bundle = buildContext(persona: persona, entities: entityCards, situation: situation, chunks: [])
            return ([], bundle)
        }

        let queryEmbedding = try await openAI.embedText(query)
        var candidateChunks = chunks

        if !entityCards.isEmpty {
            let ids = Set(entityCards.map { $0.entityId.lowercased() })
            let filtered = candidateChunks.filter { chunk in
                !Set(chunk.participants.map { $0.lowercased() }).isDisjoint(with: ids)
            }
            if !filtered.isEmpty {
                candidateChunks = filtered
            }
        }

        let preferredSources = preferredSourceKinds(for: query)
        if !preferredSources.isEmpty {
            let filtered = candidateChunks.filter { chunk in
                !Set(chunk.sourceKinds).isDisjoint(with: preferredSources)
            }
            if !filtered.isEmpty {
                candidateChunks = filtered
            }
        }

        let index = MemoryIndex(chunks: candidateChunks)
        let preliminary = index.searchWithScores(
            queryEmbedding: queryEmbedding,
            maxResults: max(30, k * 3)
        )

        var scoredChunks = preliminary
        if scoredChunks.count > k {
            let judgments = try await openAI.relevanceJudgments(query: query, chunkSummaries: scoredChunks.map { $0.chunk })
            let filtered = scoredChunks.filter { judgments[$0.chunk.id] ?? true }
            if !filtered.isEmpty {
                scoredChunks = filtered
            }
        }

        let recencyWeights = recencyMap(for: chunks)
        let final = scoredChunks.map { tuple -> (MemoryChunk, Double) in
            let similarity = normalizeCosine(tuple.similarity)
            let recency = recencyWeights[tuple.chunk.id] ?? 0.5
            let score = 0.7 * similarity + 0.3 * recency
            return (tuple.chunk, score)
        }
        .sorted { lhs, rhs in
            if lhs.1 == rhs.1 {
                return lhs.0.level < rhs.0.level
            }
            return lhs.1 > rhs.1
        }
        let selected = final.prefix(min(k, final.count)).map { $0.0 }
        let context = buildContext(persona: persona, entities: entityCards, situation: situation, chunks: selected)
        return (selected, context)
    }

    private func recencyMap(for chunks: [MemoryChunk]) -> [String: Double] {
        guard chunks.count > 1 else {
            return Dictionary(uniqueKeysWithValues: chunks.map { ($0.id, 1.0) })
        }
        let denom = Double(chunks.count - 1)
        var result: [String: Double] = [:]
        for (index, chunk) in chunks.enumerated() {
            result[chunk.id] = Double(index) / denom
        }
        return result
    }

    private func normalizeCosine(_ value: Double) -> Double {
        max(0, min(1, (value + 1) / 2))
    }

    private func preferredSourceKinds(for query: String) -> Set<SourceKind> {
        let lower = query.lowercased()
        var kinds: Set<SourceKind> = []
        if lower.contains("email") || lower.contains("inbox") {
            kinds.insert(.email)
        }
        if lower.contains("slack") || lower.contains("channel") {
            kinds.insert(.slack)
        }
        if lower.contains("text") || lower.contains("imessage") || lower.contains("sms") {
            kinds.insert(.imessage)
            kinds.insert(.whatsapp)
        }
        return kinds
    }

    private func buildContext(
        persona: PersonaCard?,
        entities: [EntityCard],
        situation: SituationCard?,
        chunks: [MemoryChunk]
    ) -> String {
        var sections: [String] = []
        sections.append("MY_PERSONA:\n\(persona?.summaryText ?? "Not provided.")")
        if entities.isEmpty {
            sections.append("ABOUT_THE_OTHER_PARTIES:\nNo entity cards provided.")
        } else {
            let details = entities.map { card in
                "- \(card.entityId): \(card.summaryText)"
            }.joined(separator: "\n")
            sections.append("ABOUT_THE_OTHER_PARTIES:\n\(details)")
        }
        sections.append("SITUATION:\n\(situation?.summaryText ?? "No situation card available.")")
        if chunks.isEmpty {
            sections.append("RELEVANT_CONTEXT:\nNo memory chunks selected.")
        } else {
            let chunkText = chunks.map { chunk in
                "=== Chunk (level \(chunk.level), id \(chunk.id)) ===\n\(chunk.summaryText)"
            }.joined(separator: "\n\n")
            sections.append("RELEVANT_CONTEXT:\n\(chunkText)")
        }
        return sections.joined(separator: "\n\n")
    }
}
