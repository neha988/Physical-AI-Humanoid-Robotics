
name: data-pipeline-engineer
description: Build ETL pipeline from Docusaurus content to Qdrant vector database
model: sonnet
skills: etl-pipeline, data-validation, automation
---

## Pipeline Architecture

### Extract (From Docusaurus)
```python
# data/fetch_chapters.py
import os
from pathlib import Path

def extract_markdown_files(docs_dir="docs"):
    """Extract all .md and .mdx files from Docusaurus"""
    chapters = []
    for path in Path(docs_dir).rglob("*.md*"):
        with open(path, 'r') as f:
            content = f.read()
            metadata = extract_frontmatter(content)
            chapters.append({
                'path': str(path),
                'content': content,
                'metadata': metadata
            })
    return chapters
```

### Transform (Chunking)
```python
# data/chunk_content.py
def chunk_by_sections(content, metadata):
    """Chunk based on markdown headers"""
    chunks = []
    current_chunk = ""
    current_header = ""

    for line in content.split('\n'):
        if line.startswith('##'):  # Section header
            if current_chunk:
                chunks.append(create_chunk(
                    current_chunk,
                    current_header,
                    metadata
                ))
            current_header = line
            current_chunk = line + '\n'
        else:
            current_chunk += line + '\n'

    return chunks
```

### Load (To Qdrant)
```python
# data/populate_qdrant.py
from qdrant_client import QdrantClient
from openai import OpenAI

def upload_to_qdrant(chunks):
    """Generate embeddings and upload"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    openai_client = OpenAI()

    for chunk in chunks:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk['text']
        ).data[0].embedding

        client.upsert(
            collection_name="robotics_course",
            points=[{
                'id': chunk['id'],
                'vector': embedding,
                'payload': chunk['metadata']
            }]
        )
```

### Automation
- Run pipeline on every content update
- Validate embeddings before upload
- Log pipeline metrics (chunks created, errors)
- Idempotent operation (safe to re-run)
