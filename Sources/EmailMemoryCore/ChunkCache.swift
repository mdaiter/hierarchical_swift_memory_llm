import Foundation

/// Actor-backed file cache for serialized `MemoryChunk` values.
public actor ChunkCache {
    private let directory: URL
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let fileManager = FileManager.default

    public init(directory: URL) {
        self.directory = directory
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    public func chunk(forKey key: String) -> MemoryChunk? {
        let url = fileURL(forKey: key)
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        do {
            let data = try Data(contentsOf: url)
            return try decoder.decode(MemoryChunk.self, from: data)
        } catch {
            return nil
        }
    }

    public func store(_ chunk: MemoryChunk, forKey key: String) {
        let url = fileURL(forKey: key)
        do {
            let data = try encoder.encode(chunk)
            try data.write(to: url, options: .atomic)
        } catch {
            // Best-effort cache; swallow errors
        }
    }

    private func fileURL(forKey key: String) -> URL {
        directory.appendingPathComponent("\(key).json")
    }
}
