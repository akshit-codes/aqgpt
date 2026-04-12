"""Runtime RAG pipeline for urbanemissions retrieval + grounded generation."""

from collections import defaultdict
import re

from aqgpt_core.rag.generator import build_generator
from aqgpt_core.rag.settings import MAX_CHUNKS_PER_URL, MAX_CONTEXT_CHUNKS, RAG_TOP_K
from aqgpt_core.rag.store import build_embedder, get_chroma_collection


class UrbanEmissionsRAG:
    def __init__(self):
        self.embedder = build_embedder()
        self.collection = get_chroma_collection()
        self.generator = build_generator()

    def chunk_count(self) -> int:
        return self.collection.count()

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """Query Chroma and deduplicate by URL."""
        query_embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 3, 24),
            include=["documents", "metadatas", "distances"],
        )

        if not results.get("ids") or not results["ids"][0]:
            return []

        url_counts: dict[str, int] = defaultdict(int)
        deduped: list[dict] = []

        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            url = metadata.get("url", "")

            if url_counts[url] >= MAX_CHUNKS_PER_URL:
                continue
            url_counts[url] += 1

            deduped.append(
                {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": metadata,
                    "distance": results["distances"][0][i],
                }
            )

            if len(deduped) >= min(top_k, MAX_CONTEXT_CHUNKS):
                break

        return deduped

    @staticmethod
    def _context_block(contexts: list[dict]) -> str:
        blocks = []
        for i, ctx in enumerate(contexts, 1):
            meta = ctx["metadata"]
            blocks.append(
                f"[Source {i}: {meta.get('title', 'Untitled')}]\n"
                f"URL: {meta.get('url', '')}\n"
                f"Category: {meta.get('category', 'General')}\n"
                f"{ctx['text']}\n"
            )
        return "\n---\n".join(blocks)

    @staticmethod
    def _aggregate_sources(contexts: list[dict]) -> list[dict]:
        """Aggregate chunk-level contexts into source-level entries keyed by URL.

        This ensures citation indices used by generation match the rendered source list.
        """
        sources: list[dict] = []
        seen_urls: dict[str, int] = {}

        for ctx in contexts:
            meta = ctx["metadata"]
            url = meta.get("url", "")
            title = meta.get("title", "Untitled")
            category = meta.get("category", "General")
            raw = (ctx.get("text") or "").strip()

            if raw.startswith(title + "\n\n"):
                raw = raw[len(title) + 2 :]

            quote = raw.strip()
            snippet = quote[:220] + ("..." if len(quote) > 220 else "")

            if url in seen_urls:
                idx = seen_urls[url]
                if quote and quote not in sources[idx]["quotes"]:
                    sources[idx]["quotes"].append(quote)
                if len(sources[idx]["snippet"]) < len(snippet):
                    sources[idx]["snippet"] = snippet
            else:
                seen_urls[url] = len(sources)
                sources.append(
                    {
                        "url": url,
                        "title": title,
                        "category": category,
                        "snippet": snippet,
                        "quotes": [quote] if quote else [],
                    }
                )

        return sources

    @staticmethod
    def _normalize_citations(answer: str, max_source: int) -> str:
        """Normalize citation tokens to [n] and clamp invalid indices to [?]."""

        def repl_source(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            if 1 <= idx <= max_source:
                return f"[{idx}]"
            return "[?]"

        normalized = answer
        # Convert (Source 3), Source 3, source 3 -> [3]
        normalized = re.sub(r"\(?\b[Ss]ource\s+(\d+)\b\)?", repl_source, normalized)

        # Clamp existing [n] citations as well
        def repl_bracket(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            if 1 <= idx <= max_source:
                return f"[{idx}]"
            return "[?]"

        normalized = re.sub(r"\[(\d+)\]", repl_bracket, normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def query(self, question: str, chat_history: list[dict] | None = None) -> tuple[str, list[dict]]:
        contexts = self.retrieve(question)
        if not contexts:
            return (
                "I couldn't find relevant information from urbanemissions.info for this query.",
                [],
            )

        sources = self._aggregate_sources(contexts)

        # Build generation contexts from the same source list rendered in UI,
        # so citation numbering remains consistent.
        generation_contexts = []
        for src in sources:
            combined = "\n\n".join(src.get("quotes", [])[:2]).strip()
            generation_contexts.append(
                {
                    "metadata": {
                        "title": src.get("title", "Untitled"),
                        "url": src.get("url", ""),
                        "category": src.get("category", "General"),
                    },
                    "text": combined or src.get("snippet", ""),
                }
            )

        answer = self.generator.generate(
            question=question,
            context_block=self._context_block(generation_contexts),
            chat_history=chat_history,
        )

        answer = self._normalize_citations(answer, max_source=len(sources))

        return answer, sources
