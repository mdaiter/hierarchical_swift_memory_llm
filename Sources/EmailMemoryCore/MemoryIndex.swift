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
        filter: SearchFilter? = nil,
        extraTraversals: Int = 0
    ) -> [MemoryChunk] {
        graphIndex.search(
            queryEmbedding: queryEmbedding,
            k: k,
            filter: filter,
            includeRecency: true,
            extraTraversals: extraTraversals
        )
    }

    /// Returns chunks plus an ASCII traversal trace for debugging.
    public func searchWithTrace(
        queryEmbedding: [Double],
        k: Int,
        filter: SearchFilter? = nil,
        extraTraversals: Int = 0
    ) -> (chunks: [MemoryChunk], trace: String) {
        let result = graphIndex.searchWithTrace(
            queryEmbedding: queryEmbedding,
            k: k,
            filter: filter,
            includeRecency: true,
            extraTraversals: extraTraversals
        )
        let traceText = result.trace.joined(separator: "\n")
        return (result.chunks, traceText)
    }
}
