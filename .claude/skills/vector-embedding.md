---
name: vector-embedding
description: Best practices for generating and managing vector embeddings using OpenAI models for RAG systems. Use when generating embeddings, optimizing embedding costs, implementing batch embedding generation, configuring similarity metrics, or debugging embedding quality.
---

# Vector Embedding Skill

## Overview
Best practices for generating, managing, and optimizing vector embeddings using OpenAI's embedding models for the RAG system. Covers embedding generation, dimensionality, similarity metrics, and cost optimization.

## Embedding Models

### OpenAI Embedding Options

| Model | Dimensions | Cost (per 1M tokens) | Best For |
|-------|------------|---------------------|----------|
| text-embedding-3-small | 512, 1536 | $0.02 | Cost-sensitive, good quality |
| text-embedding-3-large | 256, 1024, 3072 | $0.13 | Highest quality |
| text-embedding-ada-002 | 1536 | $0.10 | Legacy (still good) |

**Recommendation for Textbook**: `text-embedding-3-small` with 1536 dimensions
- Good quality-to-cost ratio
- Sufficient for educational content retrieval
- Compatible with most vector databases

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Generate embedding
response = await client.embeddings.create(
    model="text-embedding-3-small",
    input="Forward kinematics calculates end-effector position from joint angles.",
    dimensions=1536  # Optional: can reduce to 512 for even lower cost
)

embedding = response.data[0].embedding  # List[float] of length 1536
```

## Embedding Generation Patterns

### Single Text Embedding

```python
async def embed_text(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Input text (max ~8000 tokens for embedding models)
        model: OpenAI embedding model

    Returns:
        Embedding vector
    """
    response = await client.embeddings.create(
        model=model,
        input=text
    )

    return response.data[0].embedding
```

### Batch Embedding (Cost-Efficient)

```python
from typing import List
import asyncio

async def embed_batch(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100
) -> List[List[float]]:
    """
    Generate embeddings in batches for efficiency.

    OpenAI allows up to ~2000 texts per request, but we use smaller batches
    for stability and progress tracking.

    Args:
        texts: List of input texts
        batch_size: Number of texts per API request

    Returns:
        List of embedding vectors
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = await client.embeddings.create(
            model=model,
            input=batch  # Can pass list of strings
        )

        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)

        print(f"Embedded batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

    return all_embeddings
```

### Handling Long Texts

Embedding models have token limits (~8191 tokens for text-embedding-3).

**Strategy 1: Truncate** (simple but loses information)
```python
from tiktoken import encoding_for_model

def truncate_text(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to fit within token limit."""
    encoding = encoding_for_model("text-embedding-3-small")
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Truncate and decode
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
```

**Strategy 2: Smart Chunking** (recommended - see smart-chunking.md)
```python
# Break long text into chunks, embed each chunk separately
chunks = smart_chunk(long_text, max_length=500)
embeddings = await embed_batch(chunks)
# Store chunks with their embeddings in vector DB
```

## Text Preprocessing

### Cleaning Text Before Embedding

```python
import re

def preprocess_for_embedding(text: str) -> str:
    """
    Clean text before embedding generation.

    Steps:
    1. Remove excessive whitespace
    2. Normalize unicode
    3. Remove special characters (optional)
    4. Lowercase (optional - embeddings are case-aware)
    """
    # Remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)

    # Normalize unicode (e.g., different quote marks → standard quotes)
    text = text.strip()

    # Optional: Remove markdown formatting for cleaner embeddings
    # text = remove_markdown(text)

    return text
```

### Should You Remove Code/Math?

**For Educational Content**: **No**
- Code and math are semantically meaningful
- Users will query about code ("How to implement FK in Python?")
- Embeddings capture code semantics reasonably well

**Exception**: Very long code blocks might dilute semantic meaning
- Consider separating code explanation from code itself
- Embed: "This code implements forward kinematics using DH parameters"
- Store code separately in metadata

## Embedding Storage in Qdrant

### Point Structure

```python
from qdrant_client.models import PointStruct

# Create point for Qdrant
point = PointStruct(
    id=chunk_id,  # Unique integer or UUID
    vector=embedding,  # List[float]
    payload={
        "content": "Forward kinematics calculates...",
        "chapter": "chapter-2",
        "section": "kinematics",
        "keywords": ["kinematics", "robot", "forward"],
        "chunk_index": 0,  # Useful for preserving order
    }
)

await qdrant_client.upsert(
    collection_name="textbook_chunks",
    points=[point]
)
```

## Similarity Search

### Distance Metrics

**Cosine Similarity** (Recommended for text embeddings):
- Measures angle between vectors
- Range: [-1, 1] (normalized to [0, 1] in some implementations)
- Invariant to vector magnitude (good for text)
- **Use this for educational content**

```python
from qdrant_client.models import Distance, VectorParams

# Create collection with cosine distance
qdrant_client.create_collection(
    collection_name="textbook_chunks",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)
```

**Euclidean Distance** (L2):
- Geometric distance between vectors
- Sensitive to magnitude
- Less common for text embeddings

**Dot Product**:
- Unnormalized similarity
- Fast but not normalized

### Search Example

```python
async def search_similar(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.7
) -> List[dict]:
    """
    Find chunks most similar to query.

    Args:
        query: User's question
        top_k: Number of results
        score_threshold: Minimum similarity score (0-1)

    Returns:
        List of matching chunks with scores
    """
    # 1. Embed query
    query_embedding = await embed_text(query)

    # 2. Search Qdrant
    results = qdrant_client.search(
        collection_name="textbook_chunks",
        query_vector=query_embedding,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True
    )

    # 3. Format results
    return [
        {
            "content": hit.payload["content"],
            "score": hit.score,
            "chapter": hit.payload["chapter"],
            "section": hit.payload["section"]
        }
        for hit in results
    ]
```

## Quality Optimization

### Improving Retrieval Quality

**1. Include Context in Chunks**
```python
# Bad: Chunk without context
chunk = "It calculates the position."

# Good: Chunk with context
chunk = "Forward kinematics calculates the end-effector position given joint angles."
```

**2. Add Metadata for Hybrid Search**
```python
# Combine vector similarity with metadata filters
results = qdrant_client.search(
    collection_name="textbook_chunks",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="chapter", match=MatchValue(value="chapter-2")),
            FieldCondition(key="keywords", match=MatchAny(any=["kinematics"]))
        ]
    ),
    limit=5
)
```

**3. Query Expansion**
```python
async def expand_query(query: str) -> str:
    """
    Expand query with synonyms for better retrieval.

    E.g., "FK" → "forward kinematics FK"
    """
    expansions = {
        "FK": "forward kinematics",
        "IK": "inverse kinematics",
        "DOF": "degrees of freedom",
        "EE": "end-effector"
    }

    expanded = query
    for abbr, full in expansions.items():
        if abbr in query:
            expanded += f" {full}"

    return expanded

# Usage
expanded_query = await expand_query("How to compute FK?")
embedding = await embed_text(expanded_query)
```

**4. Reranking**
```python
def rerank_results(query: str, results: List[dict]) -> List[dict]:
    """
    Rerank results using additional signals beyond embedding similarity.

    Signals:
    - Keyword overlap
    - Section relevance
    - Recency (for time-sensitive content)
    """
    query_keywords = set(query.lower().split())

    for result in results:
        # Keyword overlap bonus
        result_keywords = set(result["content"].lower().split())
        keyword_overlap = len(query_keywords & result_keywords)

        # Boost score
        result["rerank_score"] = result["score"] + (keyword_overlap * 0.01)

    # Re-sort by rerank_score
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)
```

## Cost Optimization

### Embeddings Cost Calculation

```python
import tiktoken

def estimate_embedding_cost(
    texts: List[str],
    model: str = "text-embedding-3-small"
) -> float:
    """
    Estimate cost of embedding generation.

    Pricing (as of 2025):
    - text-embedding-3-small: $0.02 per 1M tokens
    - text-embedding-3-large: $0.13 per 1M tokens
    """
    encoding = tiktoken.encoding_for_model(model)

    total_tokens = sum(len(encoding.encode(text)) for text in texts)

    # Price per million tokens
    price_per_million = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10
    }

    cost = (total_tokens / 1_000_000) * price_per_million[model]

    print(f"Total tokens: {total_tokens:,}")
    print(f"Estimated cost: ${cost:.4f}")

    return cost

# Example: 4 chapters, 200 chunks, avg 400 words/chunk
texts = ["example chunk " * 400 for _ in range(200)]
estimate_embedding_cost(texts)
# Output: Total tokens: ~200,000
#         Estimated cost: $0.004
```

### Caching Strategies

**1. Cache Embeddings in Database**
```python
# Don't regenerate embeddings for unchanged content
# Store in database with content hash

import hashlib

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

async def get_or_create_embedding(text: str) -> List[float]:
    hash_key = content_hash(text)

    # Check cache
    cached = await embedding_cache.get(hash_key)
    if cached:
        return cached

    # Generate and cache
    embedding = await embed_text(text)
    await embedding_cache.set(hash_key, embedding)

    return embedding
```

**2. Reduce Dimensions (Trade Quality for Cost)**
```python
# text-embedding-3-small: 1536 dims (default) vs 512 dims
# 512 dims = 3x fewer dimensions = 3x faster storage/search

response = await client.embeddings.create(
    model="text-embedding-3-small",
    input=text,
    dimensions=512  # Reduced from 1536
)

# Slight quality loss but much faster
```

## Evaluation Metrics

### Measure Retrieval Quality

```python
def evaluate_retrieval(
    test_queries: List[dict],
    retriever: callable,
    k: int = 5
) -> dict:
    """
    Evaluate retrieval quality using test queries.

    test_queries format:
    [
        {
            "query": "What is forward kinematics?",
            "relevant_chunk_ids": [10, 23, 45]  # Ground truth
        },
        ...
    ]

    Metrics:
    - Precision@k: Fraction of retrieved chunks that are relevant
    - Recall@k: Fraction of relevant chunks that were retrieved
    - MRR (Mean Reciprocal Rank): Rank of first relevant result
    """
    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    for test_case in test_queries:
        query = test_case["query"]
        relevant_ids = set(test_case["relevant_chunk_ids"])

        # Retrieve
        results = retriever(query, top_k=k)
        retrieved_ids = [r["id"] for r in results]

        # Precision@k
        relevant_retrieved = len(set(retrieved_ids) & relevant_ids)
        precision = relevant_retrieved / k if k > 0 else 0
        precision_scores.append(precision)

        # Recall@k
        recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0
        recall_scores.append(recall)

        # MRR
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_ids:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)

    return {
        "precision@k": sum(precision_scores) / len(precision_scores),
        "recall@k": sum(recall_scores) / len(recall_scores),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks)
    }
```

## Best Practices

1. **Use batch embedding**: Process multiple texts in one API call
2. **Preprocess consistently**: Clean text the same way for embedding and search
3. **Choose appropriate dimensions**: 1536 for quality, 512 for speed/cost
4. **Store metadata**: Use hybrid search (vector + filters) for better results
5. **Cache embeddings**: Don't regenerate for unchanged content
6. **Monitor costs**: Track token usage and embedding generation
7. **Evaluate quality**: Measure precision/recall on test queries
8. **Rerank when needed**: Boost results with keyword matching
9. **Use cosine distance**: Best for text embeddings
10. **Version your embeddings**: If you change preprocessing, regenerate all embeddings

## Common Pitfalls

❌ **Not batching**: Embedding texts one-by-one → **Batch them**
❌ **Exceeding token limits**: Texts > 8000 tokens → **Chunk first**
❌ **Inconsistent preprocessing**: Embed "Hello" but search "hello " → **Normalize**
❌ **Ignoring metadata**: Pure vector search → **Add filters**
❌ **No caching**: Regenerating same embeddings → **Cache them**
❌ **Wrong distance metric**: Using Euclidean for text → **Use Cosine**
❌ **No evaluation**: Don't know if retrieval quality is good → **Measure it**
❌ **Embedding too-large chunks**: Dilutes semantic meaning → **Optimal chunk size 300-500 words**

## Example: Complete Pipeline

```python
# Complete embedding pipeline for textbook content

async def ingest_chapter_embeddings(chapter_content: str, chapter_id: str):
    """
    Full pipeline: chunk → preprocess → embed → store.
    """
    # 1. Chunk content
    from chunking import SmartChunker
    chunker = SmartChunker(chunk_size=400, overlap=50)
    chunks = chunker.chunk_chapter(chapter_content, {"chapter": chapter_id})

    print(f"Created {len(chunks)} chunks")

    # 2. Preprocess
    preprocessed = [preprocess_for_embedding(c["content"]) for c in chunks]

    # 3. Estimate cost
    cost = estimate_embedding_cost(preprocessed)
    print(f"Embedding cost: ${cost:.4f}")

    # 4. Generate embeddings (batched)
    embeddings = await embed_batch(preprocessed, batch_size=100)

    # 5. Create Qdrant points
    points = [
        PointStruct(
            id=i,
            vector=embedding,
            payload=chunk
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    # 6. Upload to Qdrant
    qdrant_client.upsert(
        collection_name="textbook_chunks",
        points=points
    )

    print(f"Uploaded {len(points)} embeddings to Qdrant")
```

This skill covers everything needed for high-quality embedding generation and retrieval!
