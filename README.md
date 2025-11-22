# EmailMemory – Omnichannel Semantic Memory Engine

EmailMemory started as a lightweight Swift package for summarizing email threads. It’s now an omnichannel semantic memory engine capable of ingesting email, Slack, SMS/iMessage, WhatsApp, notes, documents, and calendar interactions, compressing them into vectorized summaries, and retrieving them efficiently for downstream LLM workflows.

## Targets

- `EmailMemoryCore`: Core library with interaction models, OpenAI client, memory builders, entity/persona/situation cards, multi-stage retrieval, and semantic compression reporting.
- `EmailMemoryDemo`: Async executable showcasing how to ingest interactions from multiple channels, build cards and chunks, run multi-stage retrieval, and draft a reply via the OpenAI API.

## Key Concepts

- **Interactions**: Unified data model across channels (`Interaction` + `SourceKind`).
- **Semantic compression**: `OmnichannelMemoryBuilder` chunks chronologically, summarizes with the LLM, embeds, and assembles a hierarchy to avoid repeatedly streaming raw transcripts.
- **Memory cards**: `EntityCard`, `PersonaCard`, and `SituationCard` capture people/org history, the user’s tone, and rolling thread state respectively.
- **Graph-based retrieval**: `GraphMemoryIndex` maintains a lightweight k-NN graph (HNSW style) with metadata filters, greedy search, recency decay, and MMR diversity so lookups stay fast even as memories grow.
- **Multi-stage retrieval**: `MultiStageRetriever` filters by participant/channel, runs the graph index, optionally re-ranks via LLM, applies recency bias, and assembles persona/entity/situation context blocks ready for prompts.
- **Context visualization**: `ContextRetriever` and `SemanticCompressionReporter` generate human-readable context blocks, ASCII trees, and compression reports to explain what the engine selected and why.

## Demo

Run the demo with valid `OPENAI_API_KEY`:

```bash
swift run EmailMemoryDemo [--chat-model gpt-4.1-mini]
```

The demo:
1. Builds ~15k omnichannel interactions (~3k emails, 2k Slack updates, 10k quick chat messages) in parallel with live progress bars.
2. Constructs memory chunks with per-chunk caching and logs shard start/finish plus upload progress.
3. Builds persona/entity/situation cards using sample text (in parallel with progress indicators).
4. Runs multi-stage retrieval for a sample query.
5. Renders the ASCII chunk tree + unified context block.
6. Dumps the graph traversal trace (ASCII) that shows how the ANN search walked the k-NN graph.
7. Drafts an LLM reply using persona, entity insights, situation summary, and selected chunks.

Caching notes:

- The demo writes interactions to `.cache/interactions.json` and memory chunks to `.cache/memoryChunks.json`. Future runs reuse them automatically.
- Each chunk (and aggregation) is also cached individually under `.cache/chunkStore/`, so if OpenAI throttles midway through a build you can rerun without losing progress.
- Use `--chat-model gpt-4.1-mini` to fall back to the lower TPM tier; `gpt-5-mini` is the default (500k TPM).

## Usage

1. **Create interactions** from your sources and feed them to `OmnichannelMemoryBuilder`.
2. **Build entity/persona/situation cards** via their builders (these are cached summaries you refresh periodically).
3. **Store memory chunks** (e.g. in SQLite/S3) alongside embeddings; you only need to rebuild when new interactions arrive.
4. **At query time**, call `MultiStageRetriever.retrieve` (via `ContextRetriever`) with the query, persona, entity cards, situation card, and available chunks to get top-k summaries + context text.
5. **Send the assembled context** to your LLM (chat completion or function call) along with the user request.

## Customization Ideas

- Swap OpenAI endpoints with another LLM provider by adjusting `OpenAIClient`.
- Persist chunks/cards to disk or a vector DB by serializing the Codable models.
- Plug your own recency/importance scoring into `MultiStageRetriever`.
- Extend `Interaction` with additional metadata (labels, tags) and filter on those before embedding.

## Licensing

MIT License. See [LICENSE](LICENSE) for details.
