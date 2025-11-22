import Foundation

/// Represents metadata filters used during vector search.
public struct SearchFilter: Sendable {
    public var requiredParticipants: Set<String>?
    public var allowedSourceKinds: Set<SourceKind>?
    public var timeWindow: DateInterval?

    public init(
        requiredParticipants: Set<String>? = nil,
        allowedSourceKinds: Set<SourceKind>? = nil,
        timeWindow: DateInterval? = nil
    ) {
        self.requiredParticipants = requiredParticipants
        self.allowedSourceKinds = allowedSourceKinds
        self.timeWindow = timeWindow
    }

    /// Checks whether a chunk satisfies the filters.
    public func isChunkEligible(_ chunk: MemoryChunk) -> Bool {
        if let rp = requiredParticipants, !rp.isEmpty {
            let normalizedRequired = Set(rp.map { $0.lowercased() })
            let p = Set(chunk.participants.map { $0.lowercased() })
            if p.isDisjoint(with: normalizedRequired) {
                return false
            }
        }

        if let kinds = allowedSourceKinds, !kinds.isEmpty {
            let k = Set(chunk.sourceKinds)
            if k.isDisjoint(with: kinds) {
                return false
            }
        }

        if let window = timeWindow, let ts = chunk.latestTimestamp {
            if !window.contains(ts) {
                return false
            }
        }

        return true
    }
}
