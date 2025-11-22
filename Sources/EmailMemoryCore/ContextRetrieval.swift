import Foundation

/// Performs semantic retrieval over existing memory chunks.
public struct ContextRetriever: Sendable {
    public let openAI: OpenAIClient
    private let multiStage: MultiStageRetriever

    /// Creates a new retriever instance.
    public init(openAI: OpenAIClient) {
        self.openAI = openAI
        self.multiStage = MultiStageRetriever(openAI: openAI)
    }

    /// Retrieves context by delegating to the multi-stage retriever.
    public func retrieveContextBundle(
        query: String,
        persona: PersonaCard?,
        entityCards: [EntityCard],
        situation: SituationCard?,
        chunks: [MemoryChunk],
        k: Int = 8
    ) async throws -> (selectedChunks: [MemoryChunk], contextText: String) {
        let result = try await multiStage.retrieve(
            query: query,
            persona: persona,
            entityCards: entityCards,
            situation: situation,
            chunks: chunks,
            k: k
        )
        return (result.selectedChunks, result.enrichedContext)
    }

    /// Renders chunks into a textual block suitable for LLM prompts.
    public func renderContext(from chunks: [MemoryChunk]) -> String {
        guard !chunks.isEmpty else { return "No context available." }
        let sorted = chunks.enumerated().sorted { lhs, rhs in
            if lhs.element.level == rhs.element.level {
                return lhs.offset < rhs.offset
            }
            return lhs.element.level > rhs.element.level
        }
        var sections: [String] = []
        for (_, chunk) in sorted {
            var lines: [String] = []
            lines.append("=== Memory Chunk (level \(chunk.level), id \(chunk.id)) ===")
            lines.append(chunk.summaryText)
            sections.append(lines.joined(separator: "\n"))
        }
        return sections.joined(separator: "\n\n")
    }

    /// Renders an ASCII tree outlining how memory chunks relate hierarchically.
    public func renderContextTree(from chunks: [MemoryChunk]) -> String {
        guard !chunks.isEmpty else { return "No context tree available." }
        let idToSet = Dictionary(uniqueKeysWithValues: chunks.map { chunk in
            (chunk.id, Set(chunk.sourceInteractionIds))
        })
        let sortedByLevel = chunks.sorted { lhs, rhs in
            if lhs.level == rhs.level {
                return lhs.id < rhs.id
            }
            return lhs.level < rhs.level
        }

        var parentMap: [String: String?] = [:]
        for chunk in sortedByLevel {
            var bestParent: MemoryChunk?
            let childSet = idToSet[chunk.id] ?? []
            for candidate in chunks where candidate.level > chunk.level {
                guard let candidateSet = idToSet[candidate.id], childSet.isSubset(of: candidateSet) else { continue }
                if let existing = bestParent {
                    if candidate.level < existing.level {
                        bestParent = candidate
                    }
                } else {
                    bestParent = candidate
                }
            }
            parentMap[chunk.id] = bestParent?.id
        }

        var children: [String: [MemoryChunk]] = [:]
        for chunk in chunks {
            if let parentId = parentMap[chunk.id] ?? nil {
                children[parentId, default: []].append(chunk)
            }
        }

        func sortedChildren(for id: String) -> [MemoryChunk] {
            children[id, default: []].sorted { lhs, rhs in
                if lhs.level == rhs.level {
                    return lhs.id < rhs.id
                }
                return lhs.level < rhs.level
            }
        }

        var lines: [String] = []
        let roots = chunks.filter { (parentMap[$0.id] ?? nil) == nil }.sorted { lhs, rhs in
            if lhs.level == rhs.level {
                return lhs.id < rhs.id
            }
            return lhs.level > rhs.level
        }

        func dfs(chunk: MemoryChunk, prefix: String, isLast: Bool) {
            let connector: String = prefix.isEmpty ? "" : (isLast ? "└─ " : "├─ ")
            let words = chunk.summaryText.split(separator: " ").prefix(12).joined(separator: " ")
            let preview = words.isEmpty ? "(no summary)" : words
            let line = "\(prefix)\(connector)[L\(chunk.level)] \(chunk.id) · \(chunk.sourceInteractionIds.count) interactions · \(preview)"
            lines.append(line)
            let childPrefix = prefix + (isLast ? "   " : "│  ")
            let childChunks = sortedChildren(for: chunk.id)
            for (index, child) in childChunks.enumerated() {
                dfs(chunk: child, prefix: childPrefix, isLast: index == childChunks.count - 1)
            }
        }

        for (index, root) in roots.enumerated() {
            dfs(chunk: root, prefix: "", isLast: index == roots.count - 1)
        }

        return lines.joined(separator: "\n")
    }

    /// Creates a ready-to-send question prompt that includes the context and ASCII tree.
    public func makeQuestionPrompt(question: String, contextChunks: [MemoryChunk]) -> String {
        let trimmedQuestion = question.trimmingCharacters(in: .whitespacesAndNewlines)
        let contextBlock = renderContext(from: contextChunks)
        let treeBlock = renderContextTree(from: contextChunks)
        return """
        System: You answer questions about ongoing conversations using only the provided context. Be concise and cite decisions or open items.

        Linear context:
        \(contextBlock)

        Thinking view (ASCII tree):
        \(treeBlock)

        User question: \(trimmedQuestion.isEmpty ? "(no question provided)" : trimmedQuestion)
        """
    }
}
