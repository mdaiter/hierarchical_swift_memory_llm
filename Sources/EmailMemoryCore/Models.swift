import Foundation

/// Supported channels contributing to omnichannel memory.
public enum SourceKind: String, Codable, Sendable {
    case email
    case imessage
    case slack
    case whatsapp
    case note
    case doc
    case calendar
}

/// Represents an interaction across any supported channel.
public struct Interaction: Identifiable, Codable, Sendable {
    public let id: String
    public let sourceKind: SourceKind
    public let threadId: String?
    public let from: String
    public let to: [String]
    public let subjectOrTitle: String?
    public let body: String
    public let timestamp: Date

    public init(
        id: String,
        sourceKind: SourceKind,
        threadId: String?,
        from: String,
        to: [String],
        subjectOrTitle: String?,
        body: String,
        timestamp: Date
    ) {
        self.id = id
        self.sourceKind = sourceKind
        self.threadId = threadId
        self.from = from
        self.to = to
        self.subjectOrTitle = subjectOrTitle
        self.body = body
        self.timestamp = timestamp
    }
}

/// Represents a summarized chunk of interactions ready for retrieval.
public struct MemoryChunk: Identifiable, Codable, Sendable {
    public let id: String
    public let level: Int
    public let summaryText: String
    public let embedding: [Double]
    public let sourceInteractionIds: [String]
    public let participants: [String]
    public let sourceKinds: [SourceKind]

    public init(
        id: String,
        level: Int,
        summaryText: String,
        embedding: [Double],
        sourceInteractionIds: [String],
        participants: [String],
        sourceKinds: [SourceKind]
    ) {
        self.id = id
        self.level = level
        self.summaryText = summaryText
        self.embedding = embedding
        self.sourceInteractionIds = sourceInteractionIds
        self.participants = participants
        self.sourceKinds = sourceKinds
    }
}
