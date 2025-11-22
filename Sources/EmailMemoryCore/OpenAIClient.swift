import Foundation

/// Errors that may occur when communicating with the OpenAI APIs.
public enum OpenAIClientError: Error, CustomStringConvertible {
    case missingAPIKey
    case requestFailed(statusCode: Int, message: String)
    case invalidResponse

    public var description: String {
        switch self {
        case .missingAPIKey:
            return "Missing OPENAI_API_KEY environment variable."
        case let .requestFailed(statusCode, message):
            return "OpenAI request failed with status \(statusCode): \(message)"
        case .invalidResponse:
            return "OpenAI returned an invalid or empty response."
        }
    }
}

/// Lightweight client responsible for summarization, embeddings, and relevance checks.
public final class OpenAIClient: @unchecked Sendable {
    private let apiKey: String?
    private let session: URLSession
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    public init(apiKey: String? = ProcessInfo.processInfo.environment["OPENAI_API_KEY"]) {
        self.apiKey = apiKey
        self.session = URLSession(configuration: .default)
        self.encoder = JSONEncoder()
        self.decoder = JSONDecoder()
    }

    /// Summarizes a list of interactions for semantic memory.
    public func summarizeInteractions(_ interactions: [Interaction]) async throws -> String {
        guard !interactions.isEmpty else { return "No interactions provided." }
        let body = renderInteractions(interactions)
        return try await summarizeTextBlock(
            body,
            taskDescription: "Summarize the following interactions for semantic memory, keeping decisions, concerns, and unresolved items."
        )
    }

    /// Summarizes arbitrary text with the provided instruction.
    public func summarizeTextBlock(_ text: String, taskDescription: String) async throws -> String {
        let prompt = "Instruction: \(taskDescription)\n\nContent:\n\(text)"
        return try await completeChat(
            systemPrompt: "You are a semantic compression assistant. Be concise, factual, and structured.",
            userPrompt: prompt,
            temperature: 0.2
        )
    }

    /// Computes an embedding vector for the provided text.
    public func embedText(_ text: String) async throws -> [Double] {
        let payload = EmbeddingsRequest(model: "text-embedding-3-small", input: text)
        let data = try encoder.encode(payload)
        let responseData = try await performRequest(
            url: URL(string: "https://api.openai.com/v1/embeddings")!,
            body: data
        )
        let embeddingsResponse = try decoder.decode(EmbeddingsResponse.self, from: responseData)
        guard let embedding = embeddingsResponse.data.first?.embedding else {
            throw OpenAIClientError.invalidResponse
        }
        return embedding
    }

    /// Performs a generic chat completion and returns the assistant text.
    public func completeChat(systemPrompt: String, userPrompt: String, temperature: Double = 0.2) async throws -> String {
        let requestBody = ChatCompletionsRequest(
            model: "gpt-4.1-mini",
            messages: [
                ChatMessage(role: "system", content: systemPrompt),
                ChatMessage(role: "user", content: userPrompt)
            ],
            temperature: temperature
        )
        let data = try encoder.encode(requestBody)
        let responseData = try await performRequest(
            url: URL(string: "https://api.openai.com/v1/chat/completions")!,
            body: data
        )
        let completion = try decoder.decode(ChatCompletionsResponse.self, from: responseData)
        guard
            let choice = completion.choices.first,
            let content = choice.message?.content,
            !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw OpenAIClientError.invalidResponse
        }
        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Asks the LLM to judge chunk relevance to a query.
    public func relevanceJudgments(query: String, chunkSummaries: [MemoryChunk]) async throws -> [String: Bool] {
        guard !chunkSummaries.isEmpty else { return [:] }
        let descriptions = chunkSummaries.map { chunk in
            "ID: \(chunk.id)\nLevel: \(chunk.level)\nSummary: \(chunk.summaryText)"
        }.joined(separator: "\n\n")
        let userPrompt = "Query: \(query)\n\nChunks:\n\(descriptions)\n\nRespond with `ID: YES` or `ID: NO` for each chunk based on relevance."
        let response = try await completeChat(
            systemPrompt: "You judge whether semantic memories are relevant to the query. Reply with YES or NO per chunk only.",
            userPrompt: userPrompt,
            temperature: 0
        )
        var result: [String: Bool] = [:]
        response.split(separator: "\n").forEach { line in
            let parts = line.split(separator: ":")
            guard parts.count >= 2 else { return }
            let id = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
            let decision = parts[1].trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
            if decision.contains("YES") {
                result[id] = true
            } else if decision.contains("NO") {
                result[id] = false
            }
        }
        return result
    }

    private func performRequest(url: URL, body: Data) async throws -> Data {
        guard let apiKey, !apiKey.isEmpty else {
            throw OpenAIClientError.missingAPIKey
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = body
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenAIClientError.invalidResponse
        }
        guard 200..<300 ~= httpResponse.statusCode else {
            let message = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw OpenAIClientError.requestFailed(statusCode: httpResponse.statusCode, message: message)
        }
        return data
    }

    private func renderInteractions(_ interactions: [Interaction]) -> String {
        interactions.sorted { $0.timestamp < $1.timestamp }.map { interaction in
            let subject = interaction.subjectOrTitle ?? "(no subject)"
            let toList = interaction.to.joined(separator: ",")
            return "[\(interaction.sourceKind.rawValue)] \(interaction.timestamp) \(interaction.from) -> \(toList) | \(subject) | \(interaction.body)"
        }.joined(separator: "\n\n")
    }
}

// MARK: - DTOs

private struct ChatCompletionsRequest: Encodable {
    let model: String
    let messages: [ChatMessage]
    let temperature: Double
}

private struct ChatMessage: Codable {
    let role: String
    let content: String
}

private struct ChatCompletionsResponse: Decodable {
    struct Choice: Decodable {
        let message: ChatMessage?
    }

    let choices: [Choice]
}

private struct EmbeddingsRequest: Encodable {
    let model: String
    let input: String
}

private struct EmbeddingsResponse: Decodable {
    struct DataItem: Decodable {
        let embedding: [Double]
    }

    let data: [DataItem]
}
