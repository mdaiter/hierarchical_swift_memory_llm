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
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]
        func ts(_ string: String) -> Date { iso.date(from: string) ?? Date() }
        let threadId = "agentic-process-costs"
        return [
            Interaction(
                id: "email-1",
                sourceKind: .email,
                threadId: threadId,
                from: "ben@vectorpulse.ai",
                to: ["matthew@atlas.vc"],
                subjectOrTitle: "Agentic pilots and pricing",
                body: "Matthew, looping back on the agentic ops pilot. Finance needs clarity on infra costs if we keep workloads in your tenancy.",
                timestamp: ts("2024-05-01T15:04:00Z")
            ),
            Interaction(
                id: "email-2",
                sourceKind: .email,
                threadId: threadId,
                from: "matthew@atlas.vc",
                to: ["ben@vectorpulse.ai"],
                subjectOrTitle: "Re: Agentic pilots and pricing",
                body: "Thanks Ben. Infra on our side adds roughly 12% overhead but gives you SOC2 + audit trails. Happy to dig in on a call.",
                timestamp: ts("2024-05-01T16:28:00Z")
            ),
            Interaction(
                id: "email-3",
                sourceKind: .email,
                threadId: threadId,
                from: "ben@vectorpulse.ai",
                to: ["matthew@atlas.vc"],
                subjectOrTitle: "Re: Agentic pilots and pricing",
                body: "Understood. Could we cap the infra uplift at 10% if we commit to 18-month term and share anonymized telemetry?",
                timestamp: ts("2024-05-02T09:10:00Z")
            ),
            Interaction(
                id: "slack-1",
                sourceKind: .slack,
                threadId: threadId,
                from: "matthew@atlas.vc",
                to: ["ops-team@atlas"],
                subjectOrTitle: "#rev-ops",
                body: "Heads up: Ben wants clarity on automation audit logs + seat minimums. Collect last quarter's churn notes before our sync.",
                timestamp: ts("2024-05-02T12:40:00Z")
            ),
            Interaction(
                id: "slack-2",
                sourceKind: .slack,
                threadId: threadId,
                from: "ops-analyst@atlas",
                to: ["matthew@atlas.vc"],
                subjectOrTitle: "DM",
                body: "Churn reasons mostly due to lack of workflow notes. Ben's team specifically asked for runbooks last fall.",
                timestamp: ts("2024-05-02T13:05:00Z")
            ),
            Interaction(
                id: "imessage-1",
                sourceKind: .imessage,
                threadId: threadId,
                from: "ben@vectorpulse.ai",
                to: ["matthew@atlas.vc"],
                subjectOrTitle: nil,
                body: "Quick ping: ok if we pilot with 20 seats in June then ramp to 60 by August?",
                timestamp: ts("2024-05-02T14:22:00Z")
            ),
            Interaction(
                id: "imessage-2",
                sourceKind: .imessage,
                threadId: threadId,
                from: "matthew@atlas.vc",
                to: ["ben@vectorpulse.ai"],
                subjectOrTitle: nil,
                body: "20 -> 60 ramp works. Need final approval on telemetry-sharing clause before we lock the uplift cap.",
                timestamp: ts("2024-05-02T14:38:00Z")
            )
        ]
    }

    static let personaSamples: [String] = [
        "Ben — appreciate the thoughtful diligence. I prefer being direct on risks while staying collaborative on next steps.",
        "Team, let's keep language crisp and data-backed. When something is uncertain, state it plainly and propose an experiment.",
        "For exec-facing replies I like opening with the decision framing, then bulleting proof points before we talk numbers."
    ]
}
