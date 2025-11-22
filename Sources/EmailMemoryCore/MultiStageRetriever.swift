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

        let filter = SearchFilter(
            requiredParticipants: requiredParticipants(from: entityCards),
            allowedSourceKinds: preferredSources.isEmpty ? nil : preferredSources,
            timeWindow: nil
        )

        let index = MemoryIndex(chunks: candidateChunks)
        var preliminary = index.search(
            queryEmbedding: queryEmbedding,
            k: max(12, k * 3),
            filter: filter
        )

        if preliminary.count > k {
            let judgments = try await openAI.relevanceJudgments(query: query, chunkSummaries: preliminary)
            let filtered = preliminary.filter { judgments[$0.id] ?? true }
            if !filtered.isEmpty {
                preliminary = filtered
            }
        }

        let rescored = preliminary.map { chunk -> (MemoryChunk, Double) in
            let similarity = cosineSimilarity(chunk.embedding, queryEmbedding)
            let recency = recencyBoost(for: chunk)
            let score = 0.7 * similarity + 0.3 * recency
            return (chunk, score)
        }
        .sorted { lhs, rhs in
            if lhs.1 == rhs.1 {
                return lhs.0.level < rhs.0.level
            }
            return lhs.1 > rhs.1
        }
        let selected = rescored.prefix(min(k, rescored.count)).map { $0.0 }
        let context = buildContext(persona: persona, entities: entityCards, situation: situation, chunks: selected)
        return (selected, context)
    }

    private func requiredParticipants(from entityCards: [EntityCard]) -> Set<String>? {
        guard !entityCards.isEmpty else { return nil }
        return Set(entityCards.map { $0.entityId.lowercased() })
    }

    private func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        guard !a.isEmpty, a.count == b.count else { return 0 }
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

    private func recencyBoost(for chunk: MemoryChunk) -> Double {
        guard let timestamp = chunk.latestTimestamp else { return 0.5 }
        let now = Date()
        let seconds = now.timeIntervalSince(timestamp)
        let days = seconds / (60 * 60 * 24)
        let halfLife = 30.0
        let decay = exp(-log(2.0) * days / halfLife)
        return decay
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
