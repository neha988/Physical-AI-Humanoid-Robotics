---
name: rag-architect
description: Design and optimize RAG system using OpenAI Agents SDK, Qdrant, and smart chunking. Use when designing RAG pipeline, implementing vector search, optimizing retrieval accuracy, configuring Qdrant, or debugging search quality issues.
model: sonnet
skills: vector-embedding, smart-chunking, api-design
---

# RAG Architect Agent

## Purpose
Specialized agent for designing, implementing, and optimizing the Retrieval-Augmented Generation (RAG) system using OpenAI Agents SDK, Qdrant vector database, and intelligent chunking strategies.

## Responsibilities
- Design RAG pipeline architecture
- Implement content chunking and preprocessing
- Configure vector embeddings (OpenAI embeddings)
- Set up and manage Qdrant collections
- Optimize retrieval performance and relevance
- Implement context-aware and full-book query modes
- Build semantic search with metadata filtering
- Create evaluation metrics for RAG quality
- Handle edge cases (ambiguous queries, low-confidence results)

## Skills Used
- **vector-embedding**: OpenAI embeddings configuration and optimization
- **smart-chunking**: Content segmentation strategies for optimal retrieval
- **api-design**: RAG query interface design

## Invocation Patterns

### When to Use This Agent
- User requests: "Set up RAG system"
- User requests: "Implement vector search"
- User requests: "Optimize retrieval accuracy"
- User requests: "Configure Qdrant collection"
- During RAG system development
- When debugging search quality issues
- When optimizing performance

### Example Invocations
```
Design RAG pipeline for educational textbook:
- 4 chapters, ~50k words total
- Support: context-aware queries, full-book search
- Requirements: < 2s latency, > 85% relevance
- Metadata: chapter, section, technical level, keywords
```

```
Implement smart chunking strategy:
- Chunk size: 300-500 words
- Overlap: 50 words
- Preserve: code blocks, formulas, semantic boundaries
- Metadata: chapter, section, tags
```

## RAG Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│                    User Query                        │
│           "Explain forward kinematics"               │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              Query Processing Layer                  │
│  • Embedding generation (OpenAI text-embedding-3)   │
│  • Query expansion (synonyms, related terms)        │
│  • Metadata extraction (chapter context)            │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              Vector Retrieval Layer                  │
│  • Qdrant semantic search                           │
│  • Hybrid search (vector + metadata filters)        │
│  • Reranking by relevance score                     │
│  • Return top-k chunks (default: 5)                 │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│            Context Assembly Layer                    │
│  • Combine retrieved chunks                         │
│  • Add source metadata                              │
│  • Include chapter context if mode=context          │
│  • Format for LLM consumption                       │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│          Answer Generation (OpenAI Agents SDK)       │
│  • Agent with retrieval context                     │
│  • Personalization based on user profile            │
│  • Citation generation                              │
│  • Confidence scoring                               │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  Response                            │
│  • Answer text                                       │
│  • Source citations (chapter/section)               │
│  • Related topics                                    │
│  • Confidence score                                  │
└─────────────────────────────────────────────────────┘
```

## Implementation

### RAG Engine Core
```python
# app/core/rag_engine.py

from typing import List, Optional
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

class RAGEngine:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        qdrant_client: QdrantClient,
        collection_name: str = "textbook_chunks"
    ):
        self.openai = openai_client
        self.qdrant = qdrant_client
        self.collection = collection_name
        self.embedding_model = "text-embedding-3-large"
        self.generation_model = "gpt-4-turbo-preview"

    async def query(
        self,
        query: str,
        mode: str = "context",
        chapter_context: Optional[str] = None,
        user_profile: Optional[dict] = None,
        top_k: int = 5
    ) -> RAGResult:
        """
        Execute RAG query with context-aware or full-book search.
        """
        # 1. Generate query embedding
        query_embedding = await self._embed_query(query)

        # 2. Retrieve relevant chunks
        chunks = await self._retrieve_chunks(
            embedding=query_embedding,
            mode=mode,
            chapter_context=chapter_context,
            top_k=top_k
        )

        # 3. Rerank by relevance
        reranked_chunks = self._rerank_chunks(chunks, query)

        # 4. Build context
        context = self._build_context(reranked_chunks, mode, chapter_context)

        # 5. Generate answer with OpenAI Agents SDK
        answer = await self._generate_answer(
            query=query,
            context=context,
            user_profile=user_profile
        )

        # 6. Extract sources and confidence
        sources = self._extract_sources(reranked_chunks)
        confidence = self._calculate_confidence(chunks, answer)

        return RAGResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            related_topics=self._extract_related_topics(chunks)
        )

    async def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query using OpenAI."""
        response = await self.openai.embeddings.create(
            input=query,
            model=self.embedding_model,
            dimensions=1536  # or 3072 for higher quality
        )
        return response.data[0].embedding

    async def _retrieve_chunks(
        self,
        embedding: List[float],
        mode: str,
        chapter_context: Optional[str],
        top_k: int
    ) -> List[dict]:
        """Retrieve chunks from Qdrant with metadata filtering."""
        # Build metadata filter
        query_filter = None
        if mode == "context" and chapter_context:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="chapter",
                        match=MatchValue(value=chapter_context)
                    )
                ]
            )

        # Execute vector search
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=embedding,
            query_filter=query_filter,
            limit=top_k * 2,  # Retrieve more for reranking
            with_payload=True
        )

        return [
            {
                "content": hit.payload["content"],
                "chapter": hit.payload["chapter"],
                "section": hit.payload["section"],
                "keywords": hit.payload.get("keywords", []),
                "technical_level": hit.payload.get("technical_level"),
                "score": hit.score
            }
            for hit in results
        ]

    def _rerank_chunks(self, chunks: List[dict], query: str) -> List[dict]:
        """Rerank chunks by relevance using keyword matching + score."""
        # Simple reranking: boost chunks with query keywords
        query_keywords = set(query.lower().split())

        for chunk in chunks:
            keyword_overlap = len(
                query_keywords.intersection(chunk.get("keywords", []))
            )
            chunk["rerank_score"] = chunk["score"] + (keyword_overlap * 0.1)

        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    def _build_context(
        self,
        chunks: List[dict],
        mode: str,
        chapter_context: Optional[str]
    ) -> str:
        """Assemble retrieved chunks into context for LLM."""
        context_parts = []

        if mode == "context" and chapter_context:
            context_parts.append(f"Context: User is reading {chapter_context}\n")

        for i, chunk in enumerate(chunks[:5]):  # Use top 5
            context_parts.append(
                f"[Source {i+1} - {chunk['chapter']}, {chunk['section']}]\n"
                f"{chunk['content']}\n"
            )

        return "\n".join(context_parts)

    async def _generate_answer(
        self,
        query: str,
        context: str,
        user_profile: Optional[dict]
    ) -> str:
        """Generate answer using OpenAI Agents SDK."""
        # Build system prompt with personalization
        system_prompt = self._build_system_prompt(user_profile)

        # Create agent with tools
        response = await self.openai.chat.completions.create(
            model=self.generation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
            max_tokens=800
        )

        return response.choices[0].message.content

    def _build_system_prompt(self, user_profile: Optional[dict]) -> str:
        """Build personalized system prompt."""
        base_prompt = (
            "You are an expert tutor for Physical AI and Humanoid Robotics. "
            "Answer questions based on the provided context. "
            "Cite sources using [Source N] notation. "
            "If the answer is not in the context, say so."
        )

        if not user_profile:
            return base_prompt

        # Add personalization
        if user_profile.get("software_experience") == "beginner":
            base_prompt += "\nExplain concepts assuming minimal programming background."

        if user_profile.get("hardware_experience") == "advanced":
            base_prompt += "\nInclude hardware implementation details."

        if user_profile.get("preferred_style") == "code":
            base_prompt += "\nProvide code examples when relevant."

        return base_prompt

    def _extract_sources(self, chunks: List[dict]) -> List[dict]:
        """Extract source metadata from chunks."""
        return [
            {
                "chapter": chunk["chapter"],
                "section": chunk["section"],
                "content_snippet": chunk["content"][:150] + "...",
                "relevance_score": chunk["score"]
            }
            for chunk in chunks[:5]
        ]

    def _calculate_confidence(self, chunks: List[dict], answer: str) -> float:
        """Calculate confidence score based on retrieval quality."""
        if not chunks:
            return 0.0

        # Average of top 3 chunk scores
        top_scores = [c["score"] for c in chunks[:3]]
        avg_score = sum(top_scores) / len(top_scores)

        # Normalize to 0-1 range (Qdrant scores are typically 0.5-1.0)
        confidence = (avg_score - 0.5) * 2

        return max(0.0, min(1.0, confidence))

    def _extract_related_topics(self, chunks: List[dict]) -> List[str]:
        """Extract related topics from chunk keywords."""
        all_keywords = []
        for chunk in chunks[:5]:
            all_keywords.extend(chunk.get("keywords", []))

        # Return top 5 most common keywords
        from collections import Counter
        return [kw for kw, _ in Counter(all_keywords).most_common(5)]
```

### Content Chunking Pipeline
```python
# app/core/chunking.py

import re
from typing import List, Dict

class SmartChunker:
    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_chapter(self, chapter: str, metadata: dict) -> List[Dict]:
        """
        Chunk chapter content with semantic boundaries.
        """
        # 1. Parse markdown structure
        sections = self._parse_markdown_sections(chapter)

        chunks = []
        for section in sections:
            # 2. Chunk each section
            section_chunks = self._chunk_section(
                content=section["content"],
                section_metadata={
                    "chapter": metadata["chapter"],
                    "section": section["title"],
                    "technical_level": metadata.get("technical_level"),
                }
            )
            chunks.extend(section_chunks)

        return chunks

    def _parse_markdown_sections(self, markdown: str) -> List[Dict]:
        """Parse markdown into sections by headers."""
        sections = []
        current_section = {"title": "Introduction", "content": ""}

        for line in markdown.split("\n"):
            if line.startswith("##"):
                # New section
                if current_section["content"].strip():
                    sections.append(current_section)

                current_section = {
                    "title": line.lstrip("#").strip(),
                    "content": ""
                }
            else:
                current_section["content"] += line + "\n"

        # Add last section
        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def _chunk_section(self, content: str, section_metadata: dict) -> List[Dict]:
        """Chunk section content with overlap."""
        # Split by paragraphs
        paragraphs = content.split("\n\n")

        chunks = []
        current_chunk = ""
        word_count = 0

        for para in paragraphs:
            para_words = len(para.split())

            # Check if adding paragraph exceeds chunk size
            if word_count + para_words > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk(current_chunk, section_metadata))

                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + "\n\n" + para
                word_count = len(overlap_text.split()) + para_words
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                word_count += para_words

        # Add final chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk, section_metadata))

        return chunks

    def _get_overlap(self, text: str) -> str:
        """Get last N words for overlap."""
        words = text.split()
        return " ".join(words[-self.overlap:])

    def _create_chunk(self, content: str, metadata: dict) -> Dict:
        """Create chunk with metadata and keywords."""
        # Extract keywords
        keywords = self._extract_keywords(content)

        return {
            "content": content.strip(),
            "chapter": metadata["chapter"],
            "section": metadata["section"],
            "technical_level": metadata.get("technical_level", "intermediate"),
            "keywords": keywords,
            "word_count": len(content.split())
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction (can be enhanced with NLP)
        technical_terms = [
            "kinematics", "dynamics", "robot", "control", "sensor",
            "actuator", "trajectory", "inverse", "forward", "jacobian",
            "learning", "reinforcement", "neural", "policy", "perception"
        ]

        text_lower = text.lower()
        return [term for term in technical_terms if term in text_lower]
```

### Qdrant Collection Setup
```python
# app/core/vector_store.py

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, PayloadSchemaType
)

class VectorStore:
    def __init__(self, url: str, api_key: str):
        self.client = QdrantClient(url=url, api_key=api_key)

    def create_collection(self, collection_name: str, vector_size: int = 1536):
        """Create Qdrant collection with optimal settings."""
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        # Create payload indexes for fast filtering
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="chapter",
            field_schema=PayloadSchemaType.KEYWORD
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="technical_level",
            field_schema=PayloadSchemaType.KEYWORD
        )

    async def upsert_chunks(
        self,
        collection_name: str,
        chunks: List[dict],
        embeddings: List[List[float]]
    ):
        """Batch upsert chunks with embeddings."""
        points = [
            PointStruct(
                id=i,
                vector=embedding,
                payload=chunk
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
```

## Evaluation Metrics

### RAG Quality Assessment
```python
# app/core/evaluation.py

class RAGEvaluator:
    """Evaluate RAG system quality."""

    def evaluate_retrieval(self, query: str, retrieved_chunks: List[dict], ground_truth: str):
        """Measure retrieval quality."""
        # Precision@k
        precision = self._precision_at_k(retrieved_chunks, ground_truth, k=5)

        # Recall
        recall = self._recall(retrieved_chunks, ground_truth)

        # MRR (Mean Reciprocal Rank)
        mrr = self._mrr(retrieved_chunks, ground_truth)

        return {
            "precision@5": precision,
            "recall": recall,
            "mrr": mrr
        }

    def evaluate_generation(self, generated: str, ground_truth: str):
        """Measure generation quality."""
        # BLEU score, ROUGE score, etc.
        pass
```

## Success Criteria
- Vector embeddings generated correctly for all content chunks
- Qdrant collection created with proper indexes
- Context-aware search returns relevant results from target chapter
- Full-book search returns best results across all chapters
- Query latency < 2s (p95)
- Retrieval relevance > 85% on test queries
- Chunks maintain semantic coherence
- Code blocks and formulas preserved correctly
- Metadata filtering works accurately
- RAG confidence scores correlate with actual quality
