---
name: smart-chunking
description: Strategies for intelligently segmenting educational content for vector database storage and RAG retrieval while preserving semantic coherence. Use when chunking content for RAG, optimizing chunk size, preserving code blocks and formulas, implementing overlap strategies, or adding metadata to chunks.
---

# Smart Chunking Skill

## Overview
Strategies for intelligently segmenting educational content into chunks optimized for vector database storage and RAG retrieval. Ensures chunks are semantically coherent, appropriately sized, and preserve context.

## Why Chunking Matters

**Problem**: Educational chapters can be 5000+ words, but:
- Embedding models have token limits (~8000 tokens)
- Large chunks dilute semantic meaning
- RAG works best with focused, relevant pieces

**Solution**: Break content into smaller, meaningful chunks
- Ideal size: **300-500 words** (or ~400-700 tokens)
- Preserve semantic boundaries (don't split mid-concept)
- Include overlap for context continuity

## Chunking Strategies

### 1. Fixed-Size Chunking (Simple but Suboptimal)

```python
def chunk_by_words(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.

    Pros: Simple, predictable chunk sizes
    Cons: May split mid-sentence or mid-concept
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
```

**Issues**:
- Can split in the middle of a code block
- May break mathematical formulas
- Loses semantic coherence

### 2. Semantic Chunking (Recommended)

Split by natural semantic boundaries:
- **Paragraphs** for prose
- **Sections** for structured content
- **Code blocks** as complete units
- **Examples** as self-contained chunks

```python
import re
from typing import List, Dict

class SmartChunker:
    def __init__(
        self,
        target_chunk_size: int = 400,  # Target words per chunk
        overlap: int = 50,             # Overlap in words
        min_chunk_size: int = 100      # Minimum viable chunk
    ):
        self.target_size = target_chunk_size
        self.overlap = overlap
        self.min_size = min_chunk_size

    def chunk_chapter(self, markdown_content: str, metadata: dict) -> List[Dict]:
        """
        Chunk a chapter preserving semantic boundaries.

        Process:
        1. Parse markdown into sections (by ## headers)
        2. Split each section by paragraphs
        3. Combine paragraphs into target-sized chunks
        4. Preserve code blocks and formulas intact
        """
        # 1. Split by sections
        sections = self._parse_markdown_sections(markdown_content)

        all_chunks = []
        for section_title, section_content in sections:
            # 2. Chunk this section
            section_chunks = self._chunk_section(
                section_content,
                section_title,
                metadata
            )
            all_chunks.extend(section_chunks)

        return all_chunks

    def _parse_markdown_sections(self, markdown: str) -> List[tuple]:
        """
        Parse markdown into sections by ## headers.

        Returns: [(section_title, section_content), ...]
        """
        sections = []
        current_section = "Introduction"
        current_content = []

        lines = markdown.split('\n')

        for line in lines:
            # Detect section header (## Title)
            if line.startswith('##'):
                # Save previous section
                if current_content:
                    sections.append((
                        current_section,
                        '\n'.join(current_content)
                    ))

                # Start new section
                current_section = line.lstrip('#').strip()
                current_content = []
            else:
                current_content.append(line)

        # Add last section
        if current_content:
            sections.append((current_section, '\n'.join(current_content)))

        return sections

    def _chunk_section(
        self,
        content: str,
        section_title: str,
        metadata: dict
    ) -> List[Dict]:
        """
        Chunk a section preserving paragraphs and special blocks.
        """
        # Extract and protect special blocks (code, math)
        content, protected_blocks = self._extract_protected_blocks(content)

        # Split by paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = []
        current_word_count = 0

        for para in paragraphs:
            para_words = len(para.split())

            # Would adding this paragraph exceed target size?
            if current_word_count + para_words > self.target_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunk_text = self._restore_protected_blocks(chunk_text, protected_blocks)

                chunks.append(self._create_chunk(
                    content=chunk_text,
                    section=section_title,
                    metadata=metadata
                ))

                # Start new chunk with overlap
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1]]  # Keep last paragraph for overlap
                    current_word_count = len(current_chunk[0].split())
                else:
                    current_chunk = []
                    current_word_count = 0

            # Add paragraph to current chunk
            current_chunk.append(para)
            current_word_count += para_words

        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_text = self._restore_protected_blocks(chunk_text, protected_blocks)

            chunks.append(self._create_chunk(
                content=chunk_text,
                section=section_title,
                metadata=metadata
            ))

        return chunks

    def _extract_protected_blocks(self, content: str) -> tuple:
        """
        Extract code blocks and math formulas to protect them from splitting.

        Returns: (content_with_placeholders, {placeholder: original_block})
        """
        protected = {}
        placeholder_id = 0

        # Protect code blocks (```...```)
        def protect_code(match):
            nonlocal placeholder_id
            placeholder = f"__CODE_BLOCK_{placeholder_id}__"
            protected[placeholder] = match.group(0)
            placeholder_id += 1
            return placeholder

        content = re.sub(r'```[\s\S]*?```', protect_code, content)

        # Protect math blocks ($$...$$)
        def protect_math(match):
            nonlocal placeholder_id
            placeholder = f"__MATH_BLOCK_{placeholder_id}__"
            protected[placeholder] = match.group(0)
            placeholder_id += 1
            return placeholder

        content = re.sub(r'\$\$[\s\S]*?\$\$', protect_math, content)

        return content, protected

    def _restore_protected_blocks(self, content: str, protected: dict) -> str:
        """Restore protected blocks."""
        for placeholder, original in protected.items():
            content = content.replace(placeholder, original)
        return content

    def _create_chunk(self, content: str, section: str, metadata: dict) -> Dict:
        """
        Create a chunk dictionary with metadata.
        """
        # Extract keywords
        keywords = self._extract_keywords(content)

        return {
            "content": content,
            "chapter": metadata["chapter"],
            "section": section,
            "keywords": keywords,
            "word_count": len(content.split()),
            "has_code": "```" in content,
            "has_math": "$$" in content or "$" in content,
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract technical keywords from text.

        For robotics content, look for domain-specific terms.
        """
        # Domain-specific keyword list
        domain_keywords = {
            # Kinematics
            "kinematics", "forward", "inverse", "jacobian", "dh", "denavit-hartenberg",
            "end-effector", "joint", "link", "transformation", "homogeneous",

            # Dynamics
            "dynamics", "force", "torque", "momentum", "inertia", "lagrangian",
            "newton-euler",

            # Control
            "control", "pid", "feedback", "controller", "trajectory", "motion planning",
            "impedance", "admittance",

            # AI/ML
            "learning", "reinforcement", "policy", "reward", "neural network",
            "deep learning", "imitation", "supervised", "unsupervised",

            # Robotics
            "robot", "actuator", "sensor", "manipulator", "mobile", "humanoid",
            "gripper", "vision", "perception", "localization", "slam",

            # Math/CS
            "matrix", "vector", "optimization", "gradient", "algorithm",
            "differential", "integration"
        }

        text_lower = text.lower()
        found_keywords = []

        for keyword in domain_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        return found_keywords[:10]  # Limit to top 10
```

### 3. Hierarchy-Aware Chunking

Preserve document hierarchy in chunks:

```python
def chunk_with_hierarchy(content: str) -> List[Dict]:
    """
    Create chunks that preserve hierarchical context.

    Each chunk includes:
    - Current section content
    - Parent section context (breadcrumbs)
    - Section level (H2, H3, etc.)
    """
    chunks = []

    # Parse hierarchical structure
    hierarchy = parse_markdown_hierarchy(content)

    for section in hierarchy:
        # Build breadcrumb trail
        breadcrumb = " > ".join(section["path"])

        chunk = {
            "content": section["content"],
            "section_path": breadcrumb,  # E.g., "Chapter 2 > Kinematics > Forward Kinematics"
            "section_level": section["level"],  # E.g., 2 for ##, 3 for ###
            "parent_section": section["parent"]
        }

        chunks.append(chunk)

    return chunks
```

## Chunk Size Guidelines

### Optimal Chunk Sizes by Content Type

| Content Type | Optimal Size (words) | Reasoning |
|--------------|---------------------|-----------|
| Theory explanation | 400-600 | Need full context for concepts |
| Code example with explanation | 300-400 | Code + explanation together |
| Mathematical derivation | 300-500 | Complete proof/derivation |
| Exercise/problem | 200-300 | Self-contained problem |
| Definition | 100-200 | Concise, focused |

### Overlap Strategy

**Why overlap?**: Preserves context across chunk boundaries

**How much?**: **10-15% of chunk size**
- 400-word chunks → 50-word overlap
- Typically 1-2 paragraphs

```python
# Example with overlap
chunks = [
    "...end of chunk 1. This paragraph is shared.",
    "This paragraph is shared. Beginning of chunk 2..."
]
```

## Special Handling

### Code Blocks

**Rule**: Never split code blocks across chunks

```python
def _should_keep_together(self, text: str) -> bool:
    """Check if text block should not be split."""
    # Code block
    if text.startswith('```') and text.endswith('```'):
        return True

    # Math formula
    if '$$' in text:
        return True

    return False
```

### Tables and Lists

**Strategy**: Keep tables and ordered lists intact

```python
def _extract_tables(content: str) -> tuple:
    """Extract markdown tables as complete units."""
    # Detect table by pipe characters in adjacent lines
    table_pattern = r'(\|[^\n]+\|\n)+\|[^\n]+\|'

    tables = {}
    placeholder_id = 0

    def replace_table(match):
        nonlocal placeholder_id
        placeholder = f"__TABLE_{placeholder_id}__"
        tables[placeholder] = match.group(0)
        placeholder_id += 1
        return placeholder

    content = re.sub(table_pattern, replace_table, content)

    return content, tables
```

### Mathematical Formulas

**Inline math** ($...$): Can split
**Block math** ($$...$$): Keep together

```python
# Extract block math
block_math_pattern = r'\$\$[\s\S]*?\$\$'
math_blocks = re.findall(block_math_pattern, content)

# Keep each block in one chunk
```

## Chunking for Different Content Styles

### Narrative Content (Prose)

```python
# Split by paragraphs, combine to target size
paragraphs = content.split('\n\n')
chunks = combine_to_target_size(paragraphs, target=400)
```

### Technical Documentation

```python
# Split by sections/subsections
sections = parse_by_headers(content)
chunks = [create_chunk(section) for section in sections]
```

### Mixed Content (Code + Explanation)

```python
# Identify code/explanation pairs, keep together
def chunk_code_with_explanation(content):
    pattern = r'(.*?)(```[\s\S]*?```)(.*?)'
    matches = re.findall(pattern, content)

    for pre_text, code, post_text in matches:
        # Keep code with surrounding explanation
        chunk = pre_text + code + post_text
        yield chunk
```

## Metadata Enrichment

### Add Contextual Metadata to Chunks

```python
def enrich_chunk(chunk: Dict, full_chapter: str) -> Dict:
    """
    Add metadata to help with retrieval and ranking.
    """
    chunk["metadata"] = {
        # Position in document
        "chunk_index": chunk.get("index", 0),
        "total_chunks": len(all_chunks),

        # Content type flags
        "has_code": "```" in chunk["content"],
        "has_math": "$$" in chunk["content"],
        "has_diagram": "![" in chunk["content"],  # Markdown image

        # Difficulty indicators
        "technical_terms_count": count_technical_terms(chunk["content"]),
        "avg_word_length": avg_word_length(chunk["content"]),

        # Learning objective tags (if present)
        "learning_objective": extract_learning_objective(chunk["content"]),

        # Related sections
        "related_sections": find_related_sections(chunk, full_chapter)
    }

    return chunk
```

## Quality Checks

### Validate Chunks Before Embedding

```python
def validate_chunk(chunk: Dict) -> bool:
    """
    Check if chunk meets quality standards.

    Criteria:
    - Not too short (min 100 words)
    - Not too long (max 1000 words)
    - Contains meaningful content (not just headers)
    - No broken code blocks
    """
    content = chunk["content"]
    word_count = len(content.split())

    # Size checks
    if word_count < 100:
        return False

    if word_count > 1000:
        return False

    # Content quality
    if content.strip().startswith('#'):
        # Just a header, no content
        return False

    # Code block integrity
    if content.count('```') % 2 != 0:
        # Unclosed code block
        return False

    return True

# Usage
chunks = [c for c in chunks if validate_chunk(c)]
```

## Example: Complete Chunking Pipeline

```python
# Real-world example: chunk a chapter for RAG

async def process_chapter_for_rag(chapter_path: str, chapter_id: str):
    """
    Complete chunking pipeline for a textbook chapter.
    """
    # 1. Read chapter content
    with open(chapter_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. Parse frontmatter for metadata
    metadata = extract_frontmatter(content)
    metadata["chapter"] = chapter_id

    # 3. Smart chunking
    chunker = SmartChunker(
        target_chunk_size=400,
        overlap=50,
        min_chunk_size=100
    )

    chunks = chunker.chunk_chapter(content, metadata)

    print(f"Created {len(chunks)} chunks from {chapter_path}")

    # 4. Validate chunks
    valid_chunks = [c for c in chunks if validate_chunk(c)]
    print(f"{len(valid_chunks)} chunks passed validation")

    # 5. Enrich with metadata
    enriched_chunks = [enrich_chunk(c, content) for c in valid_chunks]

    # 6. Generate embeddings (see vector-embedding.md)
    from embeddings import embed_batch
    texts = [c["content"] for c in enriched_chunks]
    embeddings = await embed_batch(texts)

    # 7. Store in Qdrant
    from qdrant import store_chunks
    await store_chunks(enriched_chunks, embeddings)

    print("✅ Chapter processed and stored in vector DB")
```

## Testing Chunking Quality

### Measure Chunk Quality

```python
def analyze_chunks(chunks: List[Dict]) -> Dict:
    """
    Analyze chunk distribution and quality.
    """
    word_counts = [c["word_count"] for c in chunks]

    analysis = {
        "total_chunks": len(chunks),
        "avg_words_per_chunk": sum(word_counts) / len(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "chunks_with_code": sum(1 for c in chunks if c.get("has_code")),
        "chunks_with_math": sum(1 for c in chunks if c.get("has_math")),
        "avg_keywords_per_chunk": sum(len(c["keywords"]) for c in chunks) / len(chunks)
    }

    print("Chunk Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    return analysis
```

## Best Practices

1. **Preserve semantic boundaries**: Don't split mid-concept
2. **Keep code blocks intact**: Never break code across chunks
3. **Use appropriate overlap**: 10-15% for context continuity
4. **Target 300-500 words**: Sweet spot for retrieval quality
5. **Add rich metadata**: Keywords, section paths, content types
6. **Validate chunks**: Check size, integrity, content quality
7. **Include context**: Section titles, breadcrumbs in chunks
8. **Test retrieval**: Measure if chunking helps or hurts RAG quality
9. **Iterate based on feedback**: Monitor search quality, adjust chunking
10. **Document chunking strategy**: Know why chunks look the way they do

## Common Pitfalls

❌ **Fixed-size without semantic awareness**: Splits concepts
❌ **Too large chunks**: Dilutes relevance, exceeds token limits
❌ **Too small chunks**: Loses context, creates noise
❌ **No overlap**: Context lost at boundaries
❌ **Splitting code blocks**: Breaks syntax, unusable
❌ **Ignoring hierarchy**: Flat chunks lose structure
❌ **No metadata**: Can't filter or boost results
❌ **No validation**: Bad chunks make it to production
❌ **One-size-fits-all**: Different content needs different strategies

This skill provides everything needed for production-quality content chunking!
