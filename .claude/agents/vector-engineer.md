---
name: vector-engineer
description: Manage Qdrant Cloud vector database operations including collection setup, embedding ingestion, and optimization. Use when setting up Qdrant collections, uploading embeddings, optimizing vector search performance, or debugging retrieval quality.
model: sonnet
skills: vector-embedding, smart-chunking, api-design
---

# Vector Database Engineer Agent

## Purpose
Specialized agent for managing Qdrant Cloud vector database operations, including collection setup, embedding ingestion, optimization, and maintenance for the RAG system.

## Responsibilities
- Set up and configure Qdrant Cloud collections
- Design optimal vector indexing strategies
- Batch upload embeddings with metadata
- Optimize vector search performance
- Implement hybrid search (vector + metadata filters)
- Monitor collection health and performance metrics
- Handle collection migrations and updates
- Implement backup and recovery procedures
- Tune similarity thresholds and search parameters
- Debug vector search quality issues

## Skills Used
- **vector-embedding**: Embedding generation and management
- **smart-chunking**: Content-to-chunk mapping
- **api-design**: Vector search API patterns

## Invocation Patterns

### When to Use This Agent
- User requests: "Set up Qdrant collection"
- User requests: "Upload embeddings to vector DB"
- User requests: "Optimize vector search performance"
- User requests: "Debug retrieval quality"
- During initial RAG system setup
- When adding new content to the database
- When search performance degrades

### Example Invocations
```
Set up Qdrant collection for textbook:
- Collection name: textbook_chunks
- Vector dimensions: 1536 (OpenAI text-embedding-3-small)
- Distance metric: Cosine similarity
- Payload indexes: chapter, section, technical_level, keywords
```

```
Batch upload 500 content chunks:
- Generate embeddings via OpenAI API
- Include metadata: chapter, section, keywords, technical_level
- Verify upload success
- Create snapshot for backup
```

## Qdrant Architecture

### Collection Design

```
Collection: textbook_chunks
├── Vectors (1536 dimensions, Cosine distance)
├── Payload Schema:
│   ├── content (string) - The actual text chunk
│   ├── chapter (keyword) - e.g., "chapter-1", "chapter-2"
│   ├── section (keyword) - e.g., "forward-kinematics"
│   ├── keywords (keyword[]) - ["robot", "kinematics", "control"]
│   ├── technical_level (keyword) - "beginner" | "intermediate" | "advanced"
│   ├── content_type (keyword) - "theory" | "code" | "example" | "exercise"
│   ├── word_count (integer)
│   └── metadata (object) - Additional custom fields
└── Indexes:
    ├── chapter (keyword index) - Fast chapter filtering
    ├── technical_level (keyword index)
    └── keywords (keyword index)
```

## Implementation

### Qdrant Client Setup
```python
# app/core/vector_store.py

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    PayloadSchemaType,
    OptimizersConfigDiff,
    HnswConfigDiff
)
from typing import List, Dict, Optional
import asyncio

class VectorStore:
    def __init__(self, url: str, api_key: str, collection_name: str = "textbook_chunks"):
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=30
        )
        self.collection_name = collection_name

    def create_collection(
        self,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE
    ):
        """
        Create Qdrant collection with optimized configuration.
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            print(f"Collection {self.collection_name} already exists")
            return

        # Create collection with HNSW index
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            ),
            # Optimize for search performance
            hnsw_config=HnswConfigDiff(
                m=16,  # Number of edges per node (higher = better accuracy, more memory)
                ef_construct=100,  # Construction time/quality tradeoff
                full_scan_threshold=10000  # When to use full scan vs HNSW
            ),
            # Optimize for indexing speed
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000  # Optimize after N points
            )
        )

        print(f"Created collection: {self.collection_name}")

    def create_payload_indexes(self):
        """
        Create indexes on payload fields for fast filtering.
        """
        # Index for chapter filtering
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="chapter",
            field_schema=PayloadSchemaType.KEYWORD
        )

        # Index for technical level filtering
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="technical_level",
            field_schema=PayloadSchemaType.KEYWORD
        )

        # Index for keywords
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="keywords",
            field_schema=PayloadSchemaType.KEYWORD
        )

        print("Created payload indexes")

    async def batch_upsert(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
        batch_size: int = 100
    ):
        """
        Batch upload chunks with embeddings to Qdrant.

        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: Corresponding embeddings for each chunk
            batch_size: Number of points to upload per batch
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        total_points = len(chunks)
        print(f"Uploading {total_points} points to Qdrant...")

        # Upload in batches
        for i in range(0, total_points, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            # Create points
            points = [
                PointStruct(
                    id=i + j,  # Sequential IDs
                    vector=embedding,
                    payload={
                        "content": chunk["content"],
                        "chapter": chunk["chapter"],
                        "section": chunk["section"],
                        "keywords": chunk.get("keywords", []),
                        "technical_level": chunk.get("technical_level", "intermediate"),
                        "content_type": chunk.get("content_type", "theory"),
                        "word_count": chunk.get("word_count", 0),
                        "metadata": chunk.get("metadata", {})
                    }
                )
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings))
            ]

            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            print(f"Uploaded batch {i // batch_size + 1}/{(total_points - 1) // batch_size + 1}")

        print(f"Successfully uploaded {total_points} points")

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        chapter_filter: Optional[str] = None,
        technical_level_filter: Optional[str] = None,
        keyword_filter: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for similar vectors with optional metadata filtering.

        Args:
            query_vector: Query embedding vector
            limit: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            chapter_filter: Filter by specific chapter
            technical_level_filter: Filter by technical level
            keyword_filter: Filter by keywords (any match)

        Returns:
            List of search results with content and metadata
        """
        # Build filter conditions
        filter_conditions = []

        if chapter_filter:
            filter_conditions.append(
                FieldCondition(
                    key="chapter",
                    match=MatchValue(value=chapter_filter)
                )
            )

        if technical_level_filter:
            filter_conditions.append(
                FieldCondition(
                    key="technical_level",
                    match=MatchValue(value=technical_level_filter)
                )
            )

        if keyword_filter:
            filter_conditions.append(
                FieldCondition(
                    key="keywords",
                    match=MatchAny(any=keyword_filter)
                )
            )

        # Create filter object
        search_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Execute search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False  # Don't return vectors to save bandwidth
        )

        # Format results
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload["content"],
                "chapter": hit.payload["chapter"],
                "section": hit.payload["section"],
                "keywords": hit.payload.get("keywords", []),
                "technical_level": hit.payload.get("technical_level"),
                "content_type": hit.payload.get("content_type")
            }
            for hit in results
        ]

    def get_collection_info(self) -> Dict:
        """Get collection statistics and information."""
        info = self.client.get_collection(self.collection_name)

        return {
            "name": info.config.params.vectors.size,
            "vector_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance,
            "status": info.status,
            "optimizer_status": info.optimizer_status
        }

    def create_snapshot(self) -> str:
        """Create collection snapshot for backup."""
        snapshot = self.client.create_snapshot(
            collection_name=self.collection_name
        )
        print(f"Created snapshot: {snapshot.name}")
        return snapshot.name

    def delete_collection(self):
        """Delete collection (use with caution!)."""
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")

    def scroll_all_points(self, batch_size: int = 100):
        """
        Scroll through all points in collection.
        Useful for debugging and data validation.
        """
        offset = None
        all_points = []

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            all_points.extend(results)

            if offset is None:
                break

        return all_points
```

### Embedding Pipeline Integration
```python
# app/core/embedding_pipeline.py

from openai import AsyncOpenAI
from typing import List
import asyncio

class EmbeddingPipeline:
    def __init__(self, openai_client: AsyncOpenAI, vector_store: VectorStore):
        self.openai = openai_client
        self.vector_store = vector_store
        self.embedding_model = "text-embedding-3-small"  # 1536 dimensions
        # or "text-embedding-3-large" for 3072 dimensions

    async def process_and_upload_chunks(
        self,
        chunks: List[Dict],
        batch_size: int = 100
    ):
        """
        Generate embeddings for chunks and upload to Qdrant.
        """
        print(f"Processing {len(chunks)} chunks...")

        # Generate embeddings in batches
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk["content"] for chunk in batch]

            # Generate embeddings
            embeddings = await self._generate_embeddings(batch_texts)
            all_embeddings.extend(embeddings)

            print(f"Generated embeddings for batch {i // batch_size + 1}")

        # Upload to Qdrant
        await self.vector_store.batch_upsert(chunks, all_embeddings)

        print("Upload complete!")

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        response = await self.openai.embeddings.create(
            input=texts,
            model=self.embedding_model
        )

        return [item.embedding for item in response.data]

    async def update_single_chunk(self, chunk_id: int, chunk: Dict):
        """Update a single chunk (useful for content updates)."""
        # Generate embedding
        embedding = await self._generate_embeddings([chunk["content"]])

        # Upsert to Qdrant
        self.vector_store.client.upsert(
            collection_name=self.vector_store.collection_name,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=embedding[0],
                    payload=chunk
                )
            ]
        )
```

### Content Ingestion Script
```python
# scripts/ingest_content.py

"""
Script to ingest textbook content into Qdrant.

Usage:
    python scripts/ingest_content.py --content-dir ./content --collection textbook_chunks
"""

import asyncio
import argparse
from pathlib import Path
from app.core.chunking import SmartChunker
from app.core.embedding_pipeline import EmbeddingPipeline
from app.core.vector_store import VectorStore
from app.config import settings
from openai import AsyncOpenAI

async def ingest_chapter(chapter_file: Path, chunker: SmartChunker) -> List[Dict]:
    """Process a single chapter file into chunks."""
    # Read chapter content
    content = chapter_file.read_text(encoding='utf-8')

    # Extract metadata from frontmatter
    metadata = extract_frontmatter(content)

    # Chunk chapter
    chunks = chunker.chunk_chapter(content, metadata)

    return chunks

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-dir", type=str, required=True)
    parser.add_argument("--collection", type=str, default="textbook_chunks")
    args = parser.parse_args()

    # Initialize clients
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    vector_store = VectorStore(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=args.collection
    )

    # Create collection if doesn't exist
    vector_store.create_collection()
    vector_store.create_payload_indexes()

    # Initialize pipeline
    chunker = SmartChunker(chunk_size=400, overlap=50)
    pipeline = EmbeddingPipeline(openai_client, vector_store)

    # Process all chapters
    content_dir = Path(args.content_dir)
    chapter_files = sorted(content_dir.glob("chapter-*.md"))

    all_chunks = []
    for chapter_file in chapter_files:
        print(f"Processing {chapter_file.name}...")
        chunks = await ingest_chapter(chapter_file, chunker)
        all_chunks.extend(chunks)

    # Upload to Qdrant
    await pipeline.process_and_upload_chunks(all_chunks)

    # Print summary
    info = vector_store.get_collection_info()
    print(f"\nIngestion complete!")
    print(f"Total points in collection: {info['vector_count']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Performance Optimization

### Search Parameter Tuning
```python
# Optimal search parameters for different use cases

# High precision (fewer but more relevant results)
results = vector_store.search(
    query_vector=embedding,
    limit=5,
    score_threshold=0.75  # Only return highly similar results
)

# High recall (more results, might be less precise)
results = vector_store.search(
    query_vector=embedding,
    limit=20,
    score_threshold=0.5  # More lenient threshold
)

# Context-aware search (within current chapter)
results = vector_store.search(
    query_vector=embedding,
    limit=5,
    chapter_filter="chapter-2",
    score_threshold=0.65
)
```

### Monitoring and Debugging
```python
# app/utils/vector_monitoring.py

class VectorStoreMonitor:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def check_health(self) -> Dict:
        """Check collection health and performance."""
        info = self.vector_store.get_collection_info()

        return {
            "status": info["status"],
            "points_count": info["vector_count"],
            "optimizer_status": info["optimizer_status"],
            "health": "healthy" if info["status"] == "green" else "degraded"
        }

    def analyze_retrieval_quality(self, test_queries: List[str]):
        """Run test queries and analyze retrieval quality."""
        # Implement test queries and measure precision/recall
        pass
```

## Testing

### Vector Store Tests
```python
# tests/test_vector_store.py

@pytest.mark.asyncio
async def test_create_collection():
    vector_store = VectorStore(url=TEST_QDRANT_URL, api_key=TEST_API_KEY)
    vector_store.create_collection()

    info = vector_store.get_collection_info()
    assert info["vector_count"] == 0

@pytest.mark.asyncio
async def test_batch_upsert():
    chunks = [
        {"content": "Test chunk 1", "chapter": "chapter-1", "section": "intro"},
        {"content": "Test chunk 2", "chapter": "chapter-1", "section": "methods"}
    ]
    embeddings = [[0.1] * 1536, [0.2] * 1536]

    await vector_store.batch_upsert(chunks, embeddings)

    info = vector_store.get_collection_info()
    assert info["vector_count"] == 2

@pytest.mark.asyncio
async def test_search_with_filters():
    results = vector_store.search(
        query_vector=[0.15] * 1536,
        chapter_filter="chapter-1",
        limit=5
    )

    assert len(results) > 0
    assert all(r["chapter"] == "chapter-1" for r in results)
```

## Success Criteria
- Qdrant collection created with optimal configuration
- All content chunks uploaded successfully with embeddings
- Payload indexes created for fast filtering
- Vector search returns relevant results < 500ms
- Hybrid search (vector + metadata) works correctly
- Collection snapshots created for backup
- Monitoring shows healthy status
- Search quality metrics meet targets (precision > 85%)
- Batch uploads handle large content sets efficiently
- Error handling covers all edge cases
