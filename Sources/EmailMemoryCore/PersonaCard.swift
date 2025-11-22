import Foundation

/// Captures the user's tone and stylistic preferences.
public struct PersonaCard: Identifiable, Codable, Sendable {
    public let id: String
    public let title: String
    public let summaryText: String
    public let embedding: [Double]

    public init(id: String, title: String, summaryText: String, embedding: [Double]) {
        self.id = id
        self.title = title
        self.summaryText = summaryText
        self.embedding = embedding
    }
}

/// Builds persona cards from writing samples.
public struct PersonaCardBuilder: Sendable {
    public let openAI: OpenAIClient

    public init(openAI: OpenAIClient) {
        self.openAI = openAI
    }

    /// Summarizes a persona's tone and style.
    public func buildPersonaCard(id: String, title: String, samples: [String]) async throws -> PersonaCard {
        let sampleText = samples.isEmpty ? "" : samples.joined(separator: "\n---\n")
        let instruction = "Summarize how this person writes emails/messages: highlight tone, diction, pacing, and default sign-offs."
        let summary = try await openAI.summarizeTextBlock(sampleText.isEmpty ? "No samples provided." : sampleText, taskDescription: instruction)
        let embedding = try await openAI.embedText(summary)
        return PersonaCard(id: id, title: title, summaryText: summary, embedding: embedding)
    }
}
