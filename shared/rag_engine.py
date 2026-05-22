"""
Shared RAG Engine
Used by both OSS and Frontier bots, and the evaluation framework.
Backed by ChromaDB + sentence-transformers (all-MiniLM-L6-v2).
"""
import re
import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Lightweight RAG engine:
      - Chunks text by sentence with optional overlap
      - Embeds via sentence-transformers (local, no API cost)
      - Persists in ChromaDB on disk
      - Supports .txt and .pdf ingestion
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        db_path: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"RAGEngine ready — collection='{collection_name}', "
            f"chunks={self.collection.count()}"
        )

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 400,
        overlap_sentences: int = 1,
    ) -> list[str]:
        """Split text into overlapping sentence-based chunks."""
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks, current, current_len = [], [], 0

        for sent in sentences:
            slen = len(sent)
            if current_len + slen > chunk_size and current:
                chunks.append(" ".join(current))
                # Keep last N sentences for overlap
                current = current[-overlap_sentences:] if overlap_sentences else []
                current_len = sum(len(s) for s in current)
            current.append(sent)
            current_len += slen

        if current:
            chunks.append(" ".join(current))

        return chunks

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_text(self, text: str, source: str = "manual") -> int:
        """Chunk text and upsert into vector store. Returns chunk count."""
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            uid = hashlib.md5(f"{source}::{i}::{chunk[:80]}".encode()).hexdigest()
            ids.append(uid)
            docs.append(chunk)
            metas.append({"source": source, "chunk_index": i})

        # Skip already-existing IDs
        try:
            existing = set(self.collection.get(ids=ids)["ids"])
        except Exception:
            existing = set()

        new = [(i, d, m) for i, d, m in zip(ids, docs, metas) if i not in existing]
        if new:
            ni, nd, nm = zip(*new)
            self.collection.add(ids=list(ni), documents=list(nd), metadatas=list(nm))

        logger.info(f"add_text: {len(new)}/{len(chunks)} new chunks from '{source}'")
        return len(chunks)

    def add_file(self, filepath: str) -> int:
        """Read a .txt or .pdf file and add its content to the vector store."""
        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except ImportError:
                raise ImportError("pypdf not installed — run: pip install pypdf")
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .pdf")

        if not text.strip():
            logger.warning(f"File '{path.name}' appears empty after extraction.")
            return 0

        return self.add_text(text, source=path.name)

    def load_knowledge_base(self, kb_dir: str = "./knowledge_base") -> int:
        """Bulk-load all .txt and .pdf files from a directory."""
        kb_path = Path(kb_dir)
        if not kb_path.exists():
            logger.warning(f"Knowledge base directory not found: {kb_dir}")
            return 0

        total = 0
        for pattern in ("*.txt", "*.pdf"):
            for file in sorted(kb_path.glob(pattern)):
                try:
                    n = self.add_file(str(file))
                    total += n
                    logger.info(f"  Loaded '{file.name}' → {n} chunks")
                except Exception as e:
                    logger.error(f"  Failed to load '{file.name}': {e}")

        logger.info(f"Knowledge base load complete: {total} total chunks")
        return total

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> tuple[list[str], list[dict]]:
        """Return top-k relevant chunks and their metadata."""
        count = self.collection.count()
        if count == 0:
            return [], []

        n = min(top_k, count)
        results = self.collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        return docs, metas

    # ── Utils ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {"total_chunks": self.collection.count()}

    def clear(self):
        """Delete all documents from the collection."""
        self.collection.delete(where={"chunk_index": {"$gte": 0}})
        logger.info("Collection cleared.")
