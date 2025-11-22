import Foundation

/// Represents people or organizations summarized for quick recall.
public struct EntityCard: Identifiable, Codable, Sendable {
    public let id: String
    public let entityId: String
    public let kind: String
    public let summaryText: String
    public let embedding: [Double]

    public init(id: String, entityId: String, kind: String, summaryText: String, embedding: [Double]) {
        self.id = id
        self.entityId = entityId
        self.kind = kind
        self.summaryText = summaryText
        self.embedding = embedding
    }
}

/// Builds entity cards from all related interactions.
public struct EntityCardBuilder: Sendable {
    public let openAI: OpenAIClient

    public init(openAI: OpenAIClient) {
        self.openAI = openAI
    }

    /// Summarizes history/preferences/issues for a specific entity.
    public func buildEntityCard(entityId: String, interactions: [Interaction]) async throws -> EntityCard {
        let normalizedEntity = entityId.lowercased()
        let kind = normalizedEntity.contains("@") ? "person" : "organization"
        let relevant = interactions.filter { interaction in
            let participants = Set([interaction.from.lowercased()] + interaction.to.map { $0.lowercased() })
            return participants.contains(normalizedEntity)
        }
        let body = relevant.isEmpty ? "No direct interactions recorded." : render(interactions: relevant)
        let summary = try await openAI.summarizeTextBlock(
            body,
            taskDescription: "Summarize this participant's preferences, history with us, decisions made, and unresolved concerns."
        )
        let embedding = try await openAI.embedText(summary)
        return EntityCard(
            id: "entity:\(normalizedEntity)",
            entityId: normalizedEntity,
            kind: kind,
            summaryText: summary,
            embedding: embedding
        )
    }

    private func render(interactions: [Interaction]) -> String {
        interactions.sorted { $0.timestamp < $1.timestamp }.map { interaction in
            let subject = interaction.subjectOrTitle ?? "(no subject)"
            return "\(interaction.timestamp): [\(interaction.sourceKind.rawValue)] \(interaction.from) -> \(interaction.to.joined(separator: ",")) | \(subject) | \(interaction.body)"
        }.joined(separator: "\n\n")
    }
}
