from typing import List, Tuple, Optional
import os, io
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
import faiss
from .utils import chunk_text
from .text_gen import TextGenerator, ChatMessage

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

@dataclass
class RAGConfig:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5

class RAG:
    def __init__(self, cfg: Optional[RAGConfig] = None, text_gen: Optional[TextGenerator] = None):
        self.cfg = cfg or RAGConfig()
        self.embedder = SentenceTransformer(self.cfg.embed_model)
        self.index = None
        self.docs: List[str] = []
        self.text_gen = text_gen or TextGenerator()

    def add_text(self, text: str):
        chunks = chunk_text(text, 800, 80)
        self._add_chunks(chunks)

    def add_file(self, file_bytes: bytes, filename: str):
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".txt", ".md"]:
            text = file_bytes.decode("utf-8", errors="ignore")
            self.add_text(text)
        elif ext == ".pdf":
            if PdfReader is None:
                raise RuntimeError("pypdf not installed")
            reader = PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                try:
                    text += page.extract_text() or ""
                except Exception:
                    pass
            self.add_text(text)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _add_chunks(self, chunks: List[str]):
        embeds = self.embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeds.shape[1])
        # normalize for cosine
        faiss.normalize_L2(embeds)
        self.index.add(embeds)
        self.docs.extend(chunks)

    def query(self, question: str, top_k: Optional[int] = None) -> Tuple[str, List[Tuple[str, float]]]:
        if self.index is None:
            return "No documents in the index yet. Please add files first.", []
        top_k = top_k or self.cfg.top_k
        qv = self.embedder.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(qv)
        scores, idxs = self.index.search(qv, top_k)
        ctx = []
        pairs = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1: continue
            ctx.append(self.docs[i])
            pairs.append((self.docs[i][:200] + ("..." if len(self.docs[i])>200 else ""), float(s)))
        # compose system prompt
        system = "You are a helpful assistant that answers using the provided context. If the answer is not in the context, say you don't know."
        context_block = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(ctx)])
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=f"Answer the question using only the context below.\n\n{context_block}\n\nQuestion: {question}")
        ]
        answer = self.text_gen.generate(messages)
        return answer, pairs
