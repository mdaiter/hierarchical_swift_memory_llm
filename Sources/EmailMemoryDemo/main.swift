import Foundation
import EmailMemoryCore

@main
struct EmailMemoryDemoApp {
    static func main() async {
        let selectedModel = CLIArguments.value(for: "--chat-model") ?? "gpt-5-mini"
        print("[Config] Using chat model: \(selectedModel)")
        let client = OpenAIClient(chatModel: selectedModel)
        let personaBuilder = PersonaCardBuilder(openAI: client)
        let entityBuilder = EntityCardBuilder(openAI: client)
        let situationBuilder = SituationCardBuilder(openAI: client)
        let retriever = ContextRetriever(openAI: client)
        let compressionReporter = SemanticCompressionReporter()
        let cache = DemoCache()
        let chunkCache = ChunkCache(directory: cache.chunkStoreURL)
        let memoryBuilder = OmnichannelMemoryBuilder(
            openAI: client,
            maxConcurrentRequests: 16,
            chunkCache: chunkCache
        )

        do {
            let interactions: [Interaction]
            if let cachedInteractions = cache.loadInteractions() {
                print("[Cache] Loaded \(cachedInteractions.count) interactions from disk.")
                interactions = cachedInteractions
            } else {
                let (generated, _) = await measureAsync(label: "Generate interactions dataset") {
                    await SampleData.makeInteractions()
                }
                interactions = generated
                try? cache.saveInteractions(generated)
                print("[Cache] Saved interactions to disk for future runs.")
            }

            let cores = max(1, ProcessInfo.processInfo.processorCount)
            let totalInteractions = interactions.count
            let fingerprint = SampleData.fingerprint(for: interactions)
            let chunkReporter = ProgressReporter()
            let memoryChunks: [MemoryChunk]
            var memoryBuildTime: TimeInterval = 0
            if let cachedChunks = cache.loadChunks(fingerprint: fingerprint) {
                print("[Cache] Reusing cached memory chunks for fingerprint \(fingerprint).")
                memoryChunks = cachedChunks
            } else {
                let result = try await measureAsync(label: "Build semantic memory chunks (parallel)") {
                    let estimatedSeconds = max(0.5, Double(totalInteractions) / 800.0)
                    print(
                        String(
                            format: "[Estimate] Building %d interactions across %d shards (~%.1f s)",
                            totalInteractions,
                            cores,
                            estimatedSeconds
                        )
                    )
                print("[Chunks] Starting parallel shard build...")
                let result = try await buildSemanticMemoryParallel(
                    interactions: interactions,
                    builder: memoryBuilder,
                    shardCount: cores,
                    reporter: chunkReporter
                )
                print("[Chunks] Parallel shard build complete.")
                return result
            }
                memoryChunks = result.0
                memoryBuildTime = result.1
                try? cache.saveChunks(memoryChunks, fingerprint: fingerprint)
                print("[Cache] Saved memory chunks for fingerprint \(fingerprint).")
            }
            if memoryChunks.isEmpty {
                print("No memory chunks were produced. Provide interactions.")
                return
            }
            print(
                String(
                    format: "[Status] Chunk build produced %d nodes in %@",
                    memoryChunks.count,
                    formatDuration(memoryBuildTime)
                )
            )

            print("[Status] Building persona/entity/situation cards in parallel...")
            let cardReporter = ProgressReporter()
            let ((persona, entityCard, situation), cardsTime) = try await measureAsync(label: "Build persona/entity/situation cards (parallel)") {
                async let personaTask = withProgress(source: "Persona Card", reporter: cardReporter) {
                    try await personaBuilder.buildPersonaCard(
                        id: "persona:matthew",
                        title: "Matthew - investor style",
                        samples: SampleData.personaSamples
                    )
                }
                async let entityTask = withProgress(source: "Entity Card", reporter: cardReporter) {
                    try await entityBuilder.buildEntityCard(
                        entityId: "ben@vectorpulse.ai",
                        interactions: interactions
                    )
                }
                async let situationTask = withProgress(source: "Situation Card", reporter: cardReporter) {
                    try await situationBuilder.buildSituationCard(
                        title: "agentic-process-costs",
                        interactions: interactions
                    )
                }
                return (try await personaTask, try await entityTask, try await situationTask)
            }

            print("[Timing Summary]")
            print("  Memory chunks: \(formatDuration(memoryBuildTime))")
            print("  Persona/entity/situation cards: \(formatDuration(cardsTime))\n")
            print("[Status] Cards ready. Computing per-level counts + compression report...")

            print("Chunks per level:")
            let grouped = Dictionary(grouping: memoryChunks, by: { $0.level })
            for level in grouped.keys.sorted() {
                print("  Level \(level): \(grouped[level]?.count ?? 0)")
            }

            let compressionReport = compressionReporter.makeReport(rawInteractions: interactions, chunks: memoryChunks)
            print("\n\(compressionReport)\n")
            print("[Status] Compression summary complete. Launching retrieval...")

            let query = "Need to reply to Ben's latest email about agentic process costs and clarify pilot timeline."
            let retrievalReporter = ProgressReporter()
            await retrievalReporter.update(source: "Retrieval Pipeline", completed: 0, total: 4)
            let (selectedChunks, enrichedContext, metrics) = try await retriever.retrieveContextBundleWithMetrics(
                query: query,
                persona: persona,
                entityCards: [entityCard],
                situation: situation,
                chunks: memoryChunks,
                k: 6
            )
            await retrievalReporter.update(source: "Retrieval Pipeline", completed: 1, total: 4)
            print("[Retrieval Timing]")
            print("  Query embedding: \(formatDuration(metrics.embedTime))")
            print("  Index build: \(formatDuration(metrics.indexBuildTime))")
            print("  Graph search: \(formatDuration(metrics.graphSearchTime))")
            print("  Re-rank + filtering: \(formatDuration(metrics.rerankTime))")
            print("  Context assembly: \(formatDuration(metrics.contextAssemblyTime))")
            print("  Retrieval total: \(formatDuration(metrics.totalTime))\n")

            let asciiTree = retriever.renderContextTree(from: selectedChunks)
            await retrievalReporter.update(source: "Retrieval Pipeline", completed: 2, total: 4)
            print("Selected chunk tree:\n\(asciiTree)\n")
            print("Unified context:\n\(enrichedContext)\n")

            let systemPrompt = "You are an omnichannel assistant who drafts polished replies using the provided persona, entity insights, and situation summary."
            let userPrompt = """
            PERSONA_CARD:
            \(persona.summaryText)

            ENTITY_CARD:
            \(entityCard.summaryText)

            SITUATION:
            \(situation.summaryText)

            CONTEXT_SNIPPETS:
            \(retriever.renderContext(from: selectedChunks))

            USER REQUEST:
            Draft a reply to the incoming email, maintaining my tone, clarifying missing info if needed.
            """
            await retrievalReporter.update(source: "Retrieval Pipeline", completed: 3, total: 4)
            let draft = try await client.completeChat(systemPrompt: systemPrompt, userPrompt: userPrompt)
            await retrievalReporter.update(source: "Retrieval Pipeline", completed: 4, total: 4)
            print("Draft reply:\n\(draft)")
        } catch {
            let message = "Demo failed: \(error)\n"
            if let data = message.data(using: .utf8) {
                try? FileHandle.standardError.write(contentsOf: data)
            } else {
                print(message)
            }
        }
    }
}

private enum SampleData {
    private static let emailCount = 600
    private static let slackCount = 400
    private static let chatCount = 2000

    static func makeInteractions() async -> [Interaction] {
        let calendar = Calendar(identifier: .iso8601)
        let baseDate = calendar.date(from: DateComponents(year: 2024, month: 5, day: 1, hour: 8, minute: 0)) ?? Date()
        let baseReference = baseDate.timeIntervalSinceReferenceDate
        func timestamp(_ minutes: Int) -> Date {
            Date(timeIntervalSinceReferenceDate: baseReference + Double(minutes * 60))
        }
        let threadId = "agentic-process-costs"
        let reporter = ProgressReporter()

        return await withTaskGroup(of: [Interaction].self) { group in
            group.addTask {
                await generateEmails(count: emailCount, threadId: threadId, timestamp: timestamp, reporter: reporter)
            }
            group.addTask {
                await generateSlack(count: slackCount, threadId: threadId, timestamp: timestamp, reporter: reporter)
            }
            group.addTask {
                await generateChats(count: chatCount, threadId: threadId, timestamp: timestamp, reporter: reporter)
            }

            var combined: [Interaction] = []
            for await batch in group {
                combined.append(contentsOf: batch)
            }
            return combined.sorted { $0.timestamp < $1.timestamp }
        }
    }

    private static func generateEmails(
        count: Int,
        threadId: String,
        timestamp: @escaping (Int) -> Date,
        reporter: ProgressReporter
    ) async -> [Interaction] {
        guard count > 0 else { return [] }
        let concurrency = max(1, min(ProcessInfo.processInfo.processorCount, count))
        let chunkSize = max(1, count / concurrency)
        return await withTaskGroup(of: (Int, [Interaction]).self) { group in
            var start = 0
            while start < count {
                let end = min(start + chunkSize, count)
                let range = start..<end
                group.addTask {
                    var local: [Interaction] = []
                    local.reserveCapacity(range.count)
                    for i in range {
                        let fromMatthew = i % 2 == 0
                        let from = fromMatthew ? "matthew@atlas.vc" : "ben@vectorpulse.ai"
                        let to = fromMatthew ? ["ben@vectorpulse.ai", "ops@atlas.vc"] : ["matthew@atlas.vc"]
                        let subject = "Agentic pilot update #\(i + 1)"
                        let body = fromMatthew
                            ? "Following up on legal + telemetry alignment. Update #\(i + 1) covers capacity planning and finance guardrails."
                            : "Ben here — finance noted question #\(i + 1) about infra uplift vs. shared tenancy. Need clarity on audit logging."
                        local.append(
                            Interaction(
                                id: "email-\(i + 1)",
                                sourceKind: .email,
                                threadId: threadId,
                                from: from,
                                to: to,
                                subjectOrTitle: subject,
                                body: body,
                                timestamp: timestamp(i * 45)
                            )
                        )
                    }
                    return (range.count, local)
                }
                start = end
            }

            var combined: [Interaction] = []
            combined.reserveCapacity(count)
            var completed = 0
            for await (batchCount, batch) in group {
                combined.append(contentsOf: batch)
                completed += batchCount
                await reporter.update(source: "Emails", completed: completed, total: count)
            }
            return combined.sorted { $0.timestamp < $1.timestamp }
        }
    }

    private static func generateSlack(
        count: Int,
        threadId: String,
        timestamp: @escaping (Int) -> Date,
        reporter: ProgressReporter
    ) async -> [Interaction] {
        guard count > 0 else { return [] }
        let concurrency = max(1, min(ProcessInfo.processInfo.processorCount, count))
        let chunkSize = max(1, count / concurrency)
        return await withTaskGroup(of: (Int, [Interaction]).self) { group in
            var start = 0
            while start < count {
                let end = min(start + chunkSize, count)
                let range = start..<end
                group.addTask {
                    var local: [Interaction] = []
                    local.reserveCapacity(range.count)
                    for i in range {
                        let from = i % 3 == 0 ? "matthew@atlas.vc" : (i % 3 == 1 ? "ops-analyst@atlas" : "cfo@atlas.vc")
                        let (to, subject): ([String], String) = i % 5 == 0 ? (["ops-team@atlas"], "#rev-ops") : (["matthew@atlas.vc"], "DM")
                        let body = "Slack update \(i + 1): tracking audit coverage, billing carve-outs, and automation runbooks."
                        local.append(
                            Interaction(
                                id: "slack-\(i + 1)",
                                sourceKind: .slack,
                                threadId: threadId,
                                from: from,
                                to: to,
                                subjectOrTitle: subject,
                                body: body,
                                timestamp: timestamp(1400 + i * 5)
                            )
                        )
                    }
                    return (range.count, local)
                }
                start = end
            }

            var combined: [Interaction] = []
            combined.reserveCapacity(count)
            var completed = 0
            for await (batchCount, batch) in group {
                combined.append(contentsOf: batch)
                completed += batchCount
                await reporter.update(source: "Slack", completed: completed, total: count)
            }
            return combined.sorted { $0.timestamp < $1.timestamp }
        }
    }

    private static func generateChats(
        count: Int,
        threadId: String,
        timestamp: @escaping (Int) -> Date,
        reporter: ProgressReporter
    ) async -> [Interaction] {
        guard count > 0 else { return [] }
        let concurrency = max(1, min(ProcessInfo.processInfo.processorCount, count))
        let chunkSize = max(1, count / concurrency)
        return await withTaskGroup(of: (Int, [Interaction]).self) { group in
            var start = 0
            while start < count {
                let end = min(start + chunkSize, count)
                let range = start..<end
                group.addTask {
                    var local: [Interaction] = []
                    local.reserveCapacity(range.count)
                    for i in range {
                        let isWhatsApp = i % 2 == 0
                        let kind: SourceKind = isWhatsApp ? .whatsapp : .imessage
                        let from = isWhatsApp ? "ben@vectorpulse.ai" : "matthew@atlas.vc"
                        let to = isWhatsApp ? ["matthew@atlas.vc"] : ["ben@vectorpulse.ai"]
                        let body = isWhatsApp
                            ? "WhatsApp ping \(i + 1): confirming seat ramp + telemetry clause ack?"
                            : "iMessage \(i + 1): pushing deck to finance and legal for signature."
                        local.append(
                            Interaction(
                                id: "chat-\(i + 1)",
                                sourceKind: kind,
                                threadId: threadId,
                                from: from,
                                to: to,
                                subjectOrTitle: nil,
                                body: body,
                                timestamp: timestamp(1600 + i * 2)
                            )
                        )
                    }
                    return (range.count, local)
                }
                start = end
            }

            var combined: [Interaction] = []
            combined.reserveCapacity(count)
            var completed = 0
            for await (batchCount, batch) in group {
                combined.append(contentsOf: batch)
                completed += batchCount
                await reporter.update(source: "Chats", completed: completed, total: count)
            }
            return combined.sorted { $0.timestamp < $1.timestamp }
        }
    }

    static let personaSamples: [String] = [
        "Ben — appreciate the thoughtful diligence. I prefer being direct on risks while staying collaborative on next steps.",
        "Team, let's keep language crisp and data-backed. When something is uncertain, state it plainly and propose an experiment.",
        "For exec-facing replies I like opening with the decision framing, then bulleting proof points before we talk numbers."
    ]

    static func fingerprint(for interactions: [Interaction]) -> String {
        let firstId = interactions.first?.id ?? "none"
        let lastId = interactions.last?.id ?? "none"
        return "\(interactions.count)-\(firstId)-\(lastId)"
    }
}

private struct DemoCache {
    private let fm = FileManager.default
    private let rootURL: URL

    init() {
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let directory = cwd.appendingPathComponent(".cache", isDirectory: true)
        self.rootURL = directory
        try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    private var interactionsURL: URL {
        rootURL.appendingPathComponent("interactions.json", isDirectory: false)
    }

    private var memoryURL: URL {
        rootURL.appendingPathComponent("memoryChunks.json", isDirectory: false)
    }

    var chunkStoreURL: URL {
        let url = rootURL.appendingPathComponent("chunkStore", isDirectory: true)
        try? fm.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    func loadInteractions() -> [Interaction]? {
        guard fm.fileExists(atPath: interactionsURL.path) else { return nil }
        do {
            let data = try Data(contentsOf: interactionsURL)
            let decoded = try JSONDecoder().decode([Interaction].self, from: data)
            return decoded
        } catch {
            return nil
        }
    }

    func saveInteractions(_ interactions: [Interaction]) throws {
        let data = try JSONEncoder().encode(interactions)
        try data.write(to: interactionsURL, options: .atomic)
    }

    func loadChunks(fingerprint: String) -> [MemoryChunk]? {
        guard fm.fileExists(atPath: memoryURL.path) else { return nil }
        do {
            let data = try Data(contentsOf: memoryURL)
            let decoded = try JSONDecoder().decode(CachedChunksPayload.self, from: data)
            guard decoded.fingerprint == fingerprint else { return nil }
            return decoded.chunks
        } catch {
            return nil
        }
    }

    func saveChunks(_ chunks: [MemoryChunk], fingerprint: String) throws {
        let payload = CachedChunksPayload(fingerprint: fingerprint, chunks: chunks)
        let data = try JSONEncoder().encode(payload)
        try data.write(to: memoryURL, options: .atomic)
    }

    private struct CachedChunksPayload: Codable {
        let fingerprint: String
        let chunks: [MemoryChunk]
    }
}

private func buildSemanticMemoryParallel(
    interactions: [Interaction],
    builder: OmnichannelMemoryBuilder,
    shardCount: Int,
    reporter: ProgressReporter
) async throws -> [MemoryChunk] {
    guard !interactions.isEmpty else { return [] }
    let shards = max(1, min(shardCount, interactions.count))
    let perShard = max(1, interactions.count / shards)
    print("[Chunks] Partitioning \(interactions.count) interactions into \(shards) shards of ~\(perShard) each.")

    var slices: [[Interaction]] = []
    var index = 0
    while index < interactions.count {
        let end = min(index + perShard, interactions.count)
        slices.append(Array(interactions[index..<end]))
        index = end
    }

    print("[Chunks] Launching \(slices.count) shard tasks...")
    return try await withThrowingTaskGroup(of: [MemoryChunk].self) { group in
        for (idx, slice) in slices.enumerated() {
            let shardLabel = "Shard \(idx + 1)/\(slices.count)"
            print("[Chunks] → \(shardLabel) starting with \(slice.count) interactions")
            group.addTask {
                let chunks = try await builder.buildMemory(interactions: slice)
                print("[Chunks] ← \(shardLabel) produced \(chunks.count) chunks")
                await reporter.update(source: "Chunks", completed: idx + 1, total: slices.count)
                return chunks
            }
        }

        var combined: [MemoryChunk] = []
        for try await batch in group {
            combined.append(contentsOf: batch)
        }

        print("[Chunks] Merging \(combined.count) chunk results...")
        let sorted = combined.sorted {
            ($0.latestTimestamp ?? .distantPast) < ($1.latestTimestamp ?? .distantPast)
        }
        print("[Chunks] Merge + sort complete.")
        return sorted
    }
}

@discardableResult
private func withProgress<T>(
    source: String,
    reporter: ProgressReporter,
    block: () async throws -> T
) async rethrows -> T {
    await reporter.update(source: source, completed: 0, total: 1)
    let value = try await block()
    await reporter.update(source: source, completed: 1, total: 1)
    return value
}

private func measureSync<T>(label: String, _ block: () -> T) -> (T, TimeInterval) {
    let start = Date()
    let value = block()
    let duration = Date().timeIntervalSince(start)
    print("[Timing] \(label): \(formatDuration(duration))")
    return (value, duration)
}

private func measureAsync<T>(label: String, _ block: () async throws -> T) async rethrows -> (T, TimeInterval) {
    let start = Date()
    let value = try await block()
    let duration = Date().timeIntervalSince(start)
    print("[Timing] \(label): \(formatDuration(duration))")
    return (value, duration)
}

private func formatDuration(_ seconds: TimeInterval) -> String {
    String(format: "%.3f s", seconds)
}

private actor ProgressReporter {
    func update(source: String, completed: Int, total: Int) {
        let percent = Double(completed) / Double(total)
        let barLength = 24
        let filled = Int(percent * Double(barLength))
        let bar = String(repeating: "#", count: filled) + String(repeating: "-", count: barLength - filled)
        let line = String(format: "[%@] |%@| %3.0f%% (%d/%d)", source, bar, percent * 100, completed, total)
        print(line)
    }
}

private enum CLIArguments {
    static func value(for flag: String) -> String? {
        guard let index = CommandLine.arguments.firstIndex(of: flag),
              index + 1 < CommandLine.arguments.count else {
            return nil
        }
        return CommandLine.arguments[index + 1]
    }
}
