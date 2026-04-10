# -*- coding: utf-8 -*-
"""RAG engine: markdown ingestion, ChromaDB vector store, retrieval-augmented generation."""
from __future__ import annotations

import hashlib
import re

import chromadb
from openai import OpenAI

import config


class RAGEngine:
    def __init__(self):
        self.settings = config.load_settings()
        self._init_client()
        self._init_chroma()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_client(self):
        self.llm_client = OpenAI(
            api_key=self.settings["llm_api_key"],
            base_url=self.settings["llm_base_url"],
        )

    def _init_chroma(self):
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.embedding_fn = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )

    def reload_settings(self):
        self.settings = config.load_settings()
        self._init_client()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        batch_size = 20
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.llm_client.embeddings.create(
                input=batch,
                model=self.settings["embedding_model"],
            )
            all_embeddings.extend(item.embedding for item in resp.data)
        return all_embeddings

    # ------------------------------------------------------------------
    # Markdown splitting
    # ------------------------------------------------------------------

    def _split_markdown(self, content: str, filename: str) -> list[dict]:
        chunk_size = self.settings["chunk_size"]
        sections = re.split(r"\n(?=#{1,6}\s)", content)
        chunks: list[str] = []

        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= chunk_size:
                chunks.append(section)
                continue

            paragraphs = section.split("\n\n")
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) + 2 <= chunk_size:
                    current = f"{current}\n\n{para}" if current else para
                else:
                    if current:
                        chunks.append(current)
                    if len(para) > chunk_size:
                        self._split_long_text(para, chunk_size, chunks)
                        current = ""
                    else:
                        current = para
            if current:
                chunks.append(current)

        results = []
        for idx, chunk in enumerate(chunks):
            doc_id = hashlib.md5(f"{filename}:{idx}:{chunk[:50]}".encode()).hexdigest()
            results.append(
                {
                    "id": doc_id,
                    "content": chunk,
                    "metadata": {"source": filename, "chunk_index": idx},
                }
            )
        return results

    @staticmethod
    def _split_long_text(text: str, max_len: int, out: list[str]):
        words = text.split()
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= max_len:
                current = f"{current} {word}" if current else word
            else:
                if current:
                    out.append(current)
                current = word
        if current:
            out.append(current)

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------

    def add_document(self, content: str, filename: str) -> int:
        self.remove_document(filename)
        chunks = self._split_markdown(content, filename)
        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            metadatas=[c["metadata"] for c in chunks],
        )
        return len(chunks)

    def remove_document(self, filename: str):
        try:
            results = self.collection.get(where={"source": filename})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
        except Exception:
            pass

    def list_documents(self) -> list[dict]:
        try:
            results = self.collection.get()
            doc_map: dict[str, int] = {}
            for meta in results["metadatas"]:
                src = meta["source"]
                doc_map[src] = doc_map.get(src, 0) + 1
            return [{"filename": k, "chunks": v} for k, v in doc_map.items()]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # RAG query (streaming)
    # ------------------------------------------------------------------

    def query(self, question: str, history: list[dict] | None = None):
        context = ""
        sources: list[str] = []

        if self.collection.count() > 0:
            results = self.collection.query(
                query_texts=[question],
                n_results=min(self.settings["top_k"], self.collection.count()),
            )
            if results["documents"] and results["documents"][0]:
                parts = []
                for i, doc in enumerate(results["documents"][0]):
                    src = results["metadatas"][0][i]["source"]
                    parts.append(f"[来源: {src}]\n{doc}")
                    if src not in sources:
                        sources.append(src)
                context = "\n\n---\n\n".join(parts)

        system_prompt = "你是一个专业的知识助手，负责根据公司内部文档回答用户的问题。"
        if context:
            system_prompt += (
                "\n\n请根据以下参考资料回答用户问题。如果参考资料中没有相关信息，请如实告知。"
                f"\n\n参考资料：\n{context}"
            )
        else:
            system_prompt += (
                "\n\n当前知识库为空，请提醒用户上传文档后再提问。"
                "你仍然可以尝试用自己的知识回答。"
            )

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-10:])
        messages.append({"role": "user", "content": question})

        response = self.llm_client.chat.completions.create(
            model=self.settings["llm_model"],
            messages=messages,
            temperature=self.settings["temperature"],
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

        if sources:
            yield f"\n\n---\n📚 参考来源: {', '.join(sources)}"
