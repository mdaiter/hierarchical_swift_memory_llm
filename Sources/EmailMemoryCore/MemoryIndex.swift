import Foundation

/// Facade that uses GraphMemoryIndex under the hood.
public final class MemoryIndex: @unchecked Sendable {
    private let graphIndex: GraphMemoryIndex

    public init(chunks: [MemoryChunk], neighborCount: Int = 8) {
        self.graphIndex = GraphMemoryIndex(chunks: chunks, neighborCount: neighborCount)
    }

    /// Searches memory chunks with optional metadata filtering.
    public func search(
        queryEmbedding: [Double],
        k: Int,
        filter: SearchFilter? = nil
    ) -> [MemoryChunk] {
        graphIndex.search(
            queryEmbedding: queryEmbedding,
            k: k,
            filter: filter,
            includeRecency: true
        )
    }
}
