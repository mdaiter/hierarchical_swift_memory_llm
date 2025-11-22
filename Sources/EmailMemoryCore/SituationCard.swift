import Foundation

/// Tracks the current state of an ongoing topic or thread.
public struct SituationCard: Identifiable, Codable, Sendable {
    public let id: String
    public let title: String
    public let summaryText: String
    public let embedding: [Double]
    public let relatedInteractionIds: [String]

    public init(id: String, title: String, summaryText: String, embedding: [Double], relatedInteractionIds: [String]) {
        self.id = id
        self.title = title
        self.summaryText = summaryText
        self.embedding = embedding
        self.relatedInteractionIds = relatedInteractionIds
    }
}

/// Builds situation cards summarizing the live state of a thread or project.
public struct SituationCardBuilder: Sendable {
    public let openAI: OpenAIClient

    public init(openAI: OpenAIClient) {
        self.openAI = openAI
    }

    /// Summarizes the current state and next steps for the provided interactions.
    public func buildSituationCard(title: String, interactions: [Interaction]) async throws -> SituationCard {
        let context = interactions.sorted { $0.timestamp < $1.timestamp }.map { interaction in
            "[\(interaction.sourceKind.rawValue) @ \(interaction.timestamp)] \(interaction.from) → \(interaction.to.joined(separator: ",")): \(interaction.body)"
        }.joined(separator: "\n")
        let summary = try await openAI.summarizeTextBlock(
            context,
            taskDescription: "Provide a rolling state of this thread: goals, decisions, blockers, deadlines, and next steps."
        )
        let embedding = try await openAI.embedText(summary)
        return SituationCard(
            id: "situation:\(title.lowercased())",
            title: title,
            summaryText: summary,
            embedding: embedding,
            relatedInteractionIds: interactions.map { $0.id }
        )
    }
}
