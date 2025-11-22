import Foundation
import EmailMemoryCore

@main
struct EmailMemoryDemoApp {
    static func main() async {
        let client = OpenAIClient()
        let memoryBuilder = OmnichannelMemoryBuilder(openAI: client)
        let personaBuilder = PersonaCardBuilder(openAI: client)
        let entityBuilder = EntityCardBuilder(openAI: client)
        let situationBuilder = SituationCardBuilder(openAI: client)
        let retriever = ContextRetriever(openAI: client)
        let compressionReporter = SemanticCompressionReporter()

        let interactions = SampleData.makeInteractions()

        do {
            let memoryChunks = try await memoryBuilder.buildMemory(interactions: interactions)
            if memoryChunks.isEmpty {
                print("No memory chunks were produced. Provide interactions.")
                return
            }

            let persona = try await personaBuilder.buildPersonaCard(
                id: "persona:matthew",
                title: "Matthew - investor style",
                samples: SampleData.personaSamples
            )
            let entityCard = try await entityBuilder.buildEntityCard(
                entityId: "ben@vectorpulse.ai",
                interactions: interactions
            )
            let situation = try await situationBuilder.buildSituationCard(
                title: "agentic-process-costs",
                interactions: interactions
            )

            print("Chunks per level:")
            let grouped = Dictionary(grouping: memoryChunks, by: { $0.level })
            for level in grouped.keys.sorted() {
                print("  Level \(level): \(grouped[level]?.count ?? 0)")
            }

            let compressionReport = compressionReporter.makeReport(rawInteractions: interactions, chunks: memoryChunks)
            print("\n\(compressionReport)\n")

            let query = "Need to reply to Ben's latest email about agentic process costs and clarify pilot timeline."
            let (selectedChunks, enrichedContext) = try await retriever.retrieveContextBundle(
                query: query,
                persona: persona,
                entityCards: [entityCard],
                situation: situation,
                chunks: memoryChunks,
                k: 6
            )

            let asciiTree = retriever.renderContextTree(from: selectedChunks)
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
            let draft = try await client.completeChat(systemPrompt: systemPrompt, userPrompt: userPrompt, temperature: 0.4)
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
    static func makeInteractions() -> [Interaction] {
        let calendar = Calendar(identifier: .iso8601)
        let baseDate = calendar.date(from: DateComponents(year: 2024, month: 5, day: 1, hour: 8, minute: 0)) ?? Date()
        func timestamp(minutes offset: Int) -> Date {
            calendar.date(byAdding: .minute, value: offset, to: baseDate) ?? baseDate
        }
        let threadId = "agentic-process-costs"
        var interactions: [Interaction] = []

        // 30 emails between Matthew and Ben (and some CCs)
        for i in 0..<30 {
            let fromMatthew = i % 2 == 0
            let from = fromMatthew ? "matthew@atlas.vc" : "ben@vectorpulse.ai"
            let to = fromMatthew ? ["ben@vectorpulse.ai", "ops@atlas.vc"] : ["matthew@atlas.vc"]
            let subject = "Agentic pilot update #\(i + 1)"
            let body = fromMatthew
                ? "Following up on legal + telemetry alignment. Update #\(i + 1) covers capacity planning and finance guardrails."
                : "Ben here — finance noted question #\(i + 1) about infra uplift vs. shared tenancy. Need clarity on audit logging."
            interactions.append(
                Interaction(
                    id: "email-\(i + 1)",
                    sourceKind: .email,
                    threadId: threadId,
                    from: from,
                    to: to,
                    subjectOrTitle: subject,
                    body: body,
                    timestamp: timestamp(minutes: i * 45)
                )
            )
        }

        // 20 Slack messages in #rev-ops and DMs
        for i in 0..<20 {
            let from = i % 3 == 0 ? "matthew@atlas.vc" : (i % 3 == 1 ? "ops-analyst@atlas" : "cfo@atlas.vc")
            let to: [String]
            let subject: String
            if i % 5 == 0 {
                to = ["ops-team@atlas"]
                subject = "#rev-ops"
            } else {
                to = ["matthew@atlas.vc"]
                subject = "DM"
            }
            let body = "Slack update \(i + 1): tracking audit coverage, billing carve-outs, and automation runbooks."
            interactions.append(
                Interaction(
                    id: "slack-\(i + 1)",
                    sourceKind: .slack,
                    threadId: threadId,
                    from: from,
                    to: to,
                    subjectOrTitle: subject,
                    body: body,
                    timestamp: timestamp(minutes: 1400 + i * 5)
                )
            )
        }

        // 100 quick messages across iMessage and WhatsApp
        for i in 0..<100 {
            let isWhatsApp = i % 2 == 0
            let kind: SourceKind = isWhatsApp ? .whatsapp : .imessage
            let from = isWhatsApp ? "ben@vectorpulse.ai" : "matthew@atlas.vc"
            let to = isWhatsApp ? ["matthew@atlas.vc"] : ["ben@vectorpulse.ai"]
            let body = isWhatsApp
                ? "WhatsApp ping \(i + 1): confirming seat ramp + telemetry clause ack?"
                : "iMessage \(i + 1): pushing deck to finance and legal for signature."
            interactions.append(
                Interaction(
                    id: "chat-\(i + 1)",
                    sourceKind: kind,
                    threadId: threadId,
                    from: from,
                    to: to,
                    subjectOrTitle: nil,
                    body: body,
                    timestamp: timestamp(minutes: 1600 + i * 2)
                )
            )
        }

        return interactions.sorted { $0.timestamp < $1.timestamp }
    }

    static let personaSamples: [String] = [
        "Ben — appreciate the thoughtful diligence. I prefer being direct on risks while staying collaborative on next steps.",
        "Team, let's keep language crisp and data-backed. When something is uncertain, state it plainly and propose an experiment.",
        "For exec-facing replies I like opening with the decision framing, then bulleting proof points before we talk numbers."
    ]
}
