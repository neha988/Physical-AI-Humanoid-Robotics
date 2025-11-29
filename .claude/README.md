# Physical AI Textbook Platform - Agents & Skills

This directory contains specialized AI agents and reusable skills for building an AI-native interactive textbook platform for Physical AI and Humanoid Robotics.

## 📁 Directory Structure

```
.claude/
├── agents/                      # Specialized AI agents
│   ├── content-writer.md        # Educational content generation
│   ├── frontend-dev.md          # Docusaurus/React development
│   ├── backend-api.md           # FastAPI backend development
│   ├── rag-architect.md         # RAG system implementation
│   ├── auth-specialist.md       # Authentication & user management
│   ├── vector-engineer.md       # Qdrant vector database operations
│   ├── translator.md            # Urdu translation services
│   ├── personalization-expert.md # Content personalization
│   └── qa-specialist.md         # Testing & quality assurance
│
├── skills/                      # Reusable skills
│   ├── technical-writing.md     # Educational content patterns
│   ├── api-design.md            # RESTful API best practices
│   ├── vector-embedding.md      # Embedding generation & optimization
│   ├── smart-chunking.md        # Content chunking strategies
│   ├── multilingual-translation.md # Translation with context preservation
│   └── content-adaptation.md    # Personalization algorithms
│
└── README.md                    # This file
```

## 🤖 Agents Overview

### [Content Writer Agent](agents/content-writer.md)
**Purpose**: Generate high-quality educational content for Physical AI and Humanoid Robotics.

**Key Responsibilities**:
- Write comprehensive chapters with learning objectives
- Create code examples and practice problems
- Structure content for optimal RAG retrieval
- Add metadata for personalization

**When to Use**:
- "Generate Chapter 1: Introduction to Physical AI"
- "Create section on forward kinematics"
- "Write learning materials about reinforcement learning"

**Skills**: technical-writing, smart-chunking, content-adaptation

---

### [Frontend Developer Agent](agents/frontend-dev.md)
**Purpose**: Build and maintain the Docusaurus-based interactive textbook frontend.

**Key Responsibilities**:
- Set up Docusaurus project structure
- Create custom React components (chatbot, personalization UI)
- Implement authentication flows
- Build translation toggle and RTL support
- Deploy to GitHub Pages

**When to Use**:
- "Set up Docusaurus for the textbook"
- "Create embedded chatbot component"
- "Implement authentication UI"
- "Add Urdu translation toggle"

**Skills**: api-design, content-adaptation, technical-writing

---

### [Backend API Developer Agent](agents/backend-api.md)
**Purpose**: Build FastAPI backend for RAG, authentication, and personalization.

**Key Responsibilities**:
- Design RESTful API endpoints
- Integrate OpenAI Agents SDK for RAG
- Connect to Qdrant and Neon Postgres
- Implement Better-auth flows
- Build content personalization logic
- Create translation API with caching

**When to Use**:
- "Set up FastAPI backend"
- "Create RAG query endpoint"
- "Implement authentication API"
- "Build translation caching system"

**Skills**: api-design, vector-embedding, content-adaptation

---

### [RAG Architect Agent](agents/rag-architect.md)
**Purpose**: Design and implement the RAG system for educational Q&A.

**Key Responsibilities**:
- Design RAG pipeline architecture
- Implement smart content chunking
- Configure vector embeddings (OpenAI)
- Set up Qdrant collections
- Optimize retrieval accuracy
- Build context-aware and full-book query modes

**When to Use**:
- "Design RAG pipeline for textbook"
- "Implement vector search"
- "Optimize retrieval accuracy"
- "Configure Qdrant collection"

**Skills**: vector-embedding, smart-chunking, api-design

---

### [Auth Specialist Agent](agents/auth-specialist.md)
**Purpose**: Implement secure authentication using Better-auth and Neon Postgres.

**Key Responsibilities**:
- Set up Better-auth configuration
- Implement signup with background questionnaire
- Create login and session management
- Build JWT token flows
- Manage user profiles in Postgres
- Implement password reset and security

**When to Use**:
- "Set up authentication system"
- "Implement user signup/login"
- "Add password reset functionality"
- "Secure API endpoints"

**Skills**: api-design, content-adaptation

---

### [Vector Database Engineer Agent](agents/vector-engineer.md)
**Purpose**: Manage Qdrant Cloud operations for the RAG system.

**Key Responsibilities**:
- Set up Qdrant collections with optimal config
- Batch upload embeddings with metadata
- Optimize vector search performance
- Implement hybrid search (vector + filters)
- Monitor collection health
- Handle migrations and backups

**When to Use**:
- "Set up Qdrant collection"
- "Upload embeddings to vector DB"
- "Optimize vector search performance"
- "Debug retrieval quality"

**Skills**: vector-embedding, smart-chunking, api-design

---

### [Translator Agent](agents/translator.md)
**Purpose**: Implement Urdu translation with intelligent caching.

**Key Responsibilities**:
- Translate content using OpenAI GPT-4
- Preserve code blocks, formulas, technical terms
- Build translation caching system
- Handle RTL text formatting
- Maintain educational tone in translations
- Optimize translation costs

**When to Use**:
- "Implement Urdu translation"
- "Add translation caching"
- "Translate content while preserving code"

**Skills**: multilingual-translation, api-design, content-adaptation

---

### [Personalization Expert Agent](agents/personalization-expert.md)
**Purpose**: Implement content personalization based on user background.

**Key Responsibilities**:
- Design personalization algorithms
- Implement content variant selection
- Create adaptive learning paths
- Personalize RAG responses
- Track user engagement
- Build A/B testing framework

**When to Use**:
- "Implement content personalization"
- "Adapt content based on user background"
- "Create learning path recommendations"

**Skills**: content-adaptation, technical-writing, api-design

---

### [QA Specialist Agent](agents/qa-specialist.md)
**Purpose**: Comprehensive testing across the entire platform.

**Key Responsibilities**:
- Design test strategies (unit, integration, E2E)
- Test RAG retrieval quality
- Validate authentication flows
- Test content personalization
- Performance testing (latency, load)
- Security testing (XSS, SQL injection)
- Set up CI/CD pipelines

**When to Use**:
- "Write tests for RAG system"
- "Test authentication security"
- "Verify RAG accuracy"
- "Set up CI/CD testing"

**Skills**: api-design, vector-embedding, content-adaptation

---

## 🛠️ Skills Overview

### [Technical Writing](skills/technical-writing.md)
Patterns for writing high-quality educational content.

**Key Concepts**:
- Progressive disclosure (simple → complex)
- Multi-modal explanations (intuition, visual, technical)
- Learning objectives first (Bloom's taxonomy)
- Code-first for software learners
- Content structure templates

**Usage Example**:
```markdown
## Forward Kinematics

**Learning Objective**: Calculate end-effector position from joint angles.

**Intuition**: Think of FK as...
**Visual**: [Diagram]
**Technical**: $$f: Q \to X$$
**Code**: `def forward_kinematics():`
```

---

### [API Design](skills/api-design.md)
Best practices for RESTful API development.

**Key Concepts**:
- Resource-oriented design
- HTTP methods semantically
- Pydantic validation models
- Error handling and status codes
- Authentication with JWT
- Pagination and rate limiting

**Usage Example**:
```python
@router.get("/chapters/{chapter_id}", response_model=ChapterDetail)
async def get_chapter(
    chapter_id: str,
    user = Depends(get_current_user)
):
    chapter = await chapter_service.get_chapter(chapter_id)
    personalized = await personalization.adapt(chapter, user.profile)
    return personalized
```

---

### [Vector Embedding](skills/vector-embedding.md)
Embedding generation and optimization for RAG.

**Key Concepts**:
- OpenAI embedding models (text-embedding-3-small/large)
- Batch embedding for cost efficiency
- Cosine similarity for text
- Qdrant integration
- Quality evaluation (precision, recall)

**Usage Example**:
```python
# Batch embedding
embeddings = await embed_batch(chunks, batch_size=100)

# Search
results = qdrant.search(
    query_vector=query_embedding,
    limit=5,
    score_threshold=0.7
)
```

---

### [Smart Chunking](skills/smart-chunking.md)
Content segmentation strategies for RAG.

**Key Concepts**:
- Semantic chunking (preserve boundaries)
- Optimal chunk size: 300-500 words
- Overlap for context: 10-15%
- Preserve code blocks and formulas
- Metadata enrichment

**Usage Example**:
```python
chunker = SmartChunker(target_chunk_size=400, overlap=50)
chunks = chunker.chunk_chapter(content, metadata)

# Each chunk has:
# - content, chapter, section, keywords
# - has_code, has_math flags
```

---

### [Multilingual Translation](skills/multilingual-translation.md)
Translation with context preservation.

**Key Concepts**:
- Preserve code, math, technical terms
- Educational-specific prompts
- Database-backed caching
- RTL (right-to-left) handling
- Quality validation

**Usage Example**:
```python
translator = EducationalTranslator(openai_client)

translated = await translator.translate_content(
    content=chapter_text,
    target_lang="ur",
    preserve_code=True,
    preserve_math=True
)
```

---

### [Content Adaptation](skills/content-adaptation.md)
Personalization algorithms for educational content.

**Key Concepts**:
- User profile dimensions (SW/HW experience, math, style)
- Content variant selection
- Difficulty adjustment
- Learning style adaptation (visual, code, theory)
- Adaptive learning paths

**Usage Example**:
```python
selector = ContentSelector()
focus = selector.select_primary_focus(user_profile)  # sw-focused, hw-focused, etc.

personalized = selector.select_content_sections(
    all_sections=content_variants,
    profile=user_profile
)
```

---

## 🚀 Quick Start Guide

### 1. Generate Educational Content

```bash
# Use Content Writer Agent
Task: "Generate Chapter 1: Introduction to Physical AI
- Cover: embodied AI, sensor-motor loops, applications
- Target: Mixed audience (software + hardware)
- Include: 5 code examples, 3 exercises, glossary"
```

The agent will use:
- **technical-writing** skill for pedagogical structure
- **smart-chunking** skill to prepare for RAG
- **content-adaptation** skill for multi-level variants

### 2. Set Up RAG System

```bash
# Use RAG Architect Agent
Task: "Design RAG pipeline for 4-chapter textbook
- Chunk size: 400 words, 50-word overlap
- Embeddings: OpenAI text-embedding-3-small
- Vector DB: Qdrant Cloud
- Support: context-aware and full-book search"
```

The agent will use:
- **vector-embedding** skill for embedding generation
- **smart-chunking** skill for content segmentation
- **api-design** skill for query endpoint

### 3. Build Backend API

```bash
# Use Backend API Developer Agent
Task: "Create FastAPI backend with:
- RAG query endpoint
- User authentication (Better-auth)
- Translation API
- Personalization endpoint"
```

The agent will use:
- **api-design** skill for endpoint patterns
- **vector-embedding** skill for RAG integration
- **content-adaptation** skill for personalization

### 4. Implement Frontend

```bash
# Use Frontend Developer Agent
Task: "Set up Docusaurus with:
- Embedded chatbot (collapsible sidebar)
- Authentication UI (signup with questionnaire)
- Urdu translation toggle
- Personalized content rendering"
```

The agent will use:
- **api-design** skill for API integration
- **content-adaptation** skill for UI personalization
- **technical-writing** skill for component docs

### 5. Add Translation

```bash
# Use Translator Agent
Task: "Implement Urdu translation:
- Preserve code blocks and formulas
- Cache translations in Postgres
- Support RTL layout
- Maintain educational quality"
```

The agent will use:
- **multilingual-translation** skill for translation logic
- **api-design** skill for caching API
- **content-adaptation** skill for quality preservation

### 6. Test Everything

```bash
# Use QA Specialist Agent
Task: "Create comprehensive test suite:
- Unit tests: chunking, embeddings, personalization
- Integration tests: RAG pipeline, auth flows
- E2E tests: user journey (signup → read → chatbot → translate)
- Performance tests: RAG latency < 2s"
```

The agent will use:
- **api-design** skill for API contract testing
- **vector-embedding** skill for RAG quality tests
- **content-adaptation** skill for personalization tests

---

## 🎯 Common Workflows

### Workflow 1: Add New Chapter

1. **Content Writer**: Generate chapter content
2. **RAG Architect**: Chunk and embed content
3. **Vector Engineer**: Upload to Qdrant
4. **Translator**: Translate to Urdu and cache
5. **QA Specialist**: Test retrieval quality

### Workflow 2: Improve RAG Accuracy

1. **QA Specialist**: Measure current precision/recall
2. **RAG Architect**: Optimize chunking strategy
3. **Vector Engineer**: Adjust similarity thresholds
4. **Content Writer**: Improve content structure
5. **QA Specialist**: Validate improvements

### Workflow 3: Add New Language

1. **Translator**: Extend translation service
2. **Frontend Developer**: Add language toggle UI
3. **Backend API**: Update translation endpoint
4. **QA Specialist**: Test translation quality

### Workflow 4: Personalize New Content Type

1. **Content Writer**: Create content variants
2. **Personalization Expert**: Define selection logic
3. **Backend API**: Update personalization endpoint
4. **Frontend Developer**: Render variants
5. **QA Specialist**: A/B test effectiveness

---

## 📊 Agent-Skill Matrix

| Agent | Primary Skills | Secondary Skills |
|-------|---------------|------------------|
| Content Writer | technical-writing, smart-chunking | content-adaptation |
| Frontend Dev | api-design | content-adaptation, technical-writing |
| Backend API | api-design, vector-embedding | content-adaptation |
| RAG Architect | vector-embedding, smart-chunking | api-design |
| Auth Specialist | api-design | content-adaptation |
| Vector Engineer | vector-embedding, smart-chunking | api-design |
| Translator | multilingual-translation | api-design, content-adaptation |
| Personalization | content-adaptation | technical-writing, api-design |
| QA Specialist | api-design, vector-embedding | content-adaptation |

---

## 💡 Tips for Effective Use

### 1. **Combine Agents for Complex Tasks**
Don't use agents in isolation. For example:
- Content Writer → RAG Architect → Vector Engineer (content pipeline)
- Backend API → Auth Specialist (secure endpoints)
- Translator → Personalization Expert (localized personalization)

### 2. **Refer to Skills for Patterns**
When an agent needs to implement something:
- Check the relevant skill for best practices
- Follow established patterns
- Copy code examples and adapt

### 3. **Iterate Based on Metrics**
- Use QA Specialist to measure quality
- Iterate on chunking, embedding, personalization
- Track improvements quantitatively

### 4. **Start Simple, Add Complexity**
- Begin with basic RAG (no personalization)
- Add authentication
- Add personalization
- Add translation
- Test and optimize each layer

### 5. **Document Decisions**
Each agent should:
- Document why choices were made
- Note trade-offs considered
- Explain optimization rationale

---

## 🔧 Technical Stack Reference

### Frontend
- **Framework**: Docusaurus (React)
- **Language**: TypeScript
- **State**: Context API / Zustand
- **Styling**: CSS Modules
- **Deployment**: GitHub Pages

### Backend
- **Framework**: FastAPI (Python)
- **Language**: Python 3.11+
- **Validation**: Pydantic
- **Authentication**: Better-auth (JWT)

### Databases
- **Vector DB**: Qdrant Cloud
- **Relational DB**: Neon Serverless Postgres
- **Caching**: Redis (optional) or DB-backed

### AI/ML
- **LLM**: OpenAI GPT-4 Turbo
- **Embeddings**: OpenAI text-embedding-3-small
- **RAG**: OpenAI Agents SDK

### DevOps
- **CI/CD**: GitHub Actions
- **Testing**: pytest, Playwright
- **Deployment**: GitHub Pages (frontend), Railway/Render (backend)

---

## 📚 Additional Resources

### Learning Materials
- [Docusaurus Documentation](https://docusaurus.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [OpenAI API Reference](https://platform.openai.com/docs/)

### Best Practices Guides
- See individual agent files for detailed implementations
- See individual skill files for pattern libraries
- Check code examples in each file for copy-paste templates

---

## 🤝 Contributing

### Adding New Agents
1. Create `agents/your-agent.md`
2. Follow the template:
   - Purpose and responsibilities
   - Skills used
   - Invocation patterns
   - Implementation examples
   - Success criteria
3. Update this README with agent overview
4. Update agent-skill matrix

### Adding New Skills
1. Create `skills/your-skill.md`
2. Follow the template:
   - Overview and core principles
   - Code patterns and examples
   - Best practices
   - Common pitfalls
3. Update this README with skill overview
4. Reference skill from relevant agents

---

## 📝 License

This project structure and documentation is designed for the Physical AI & Humanoid Robotics textbook platform.

---

## 🎓 Project Context

**Goal**: Build an AI-native interactive textbook platform that:
- Generates high-quality educational content
- Provides RAG-powered Q&A chatbot
- Personalizes content based on user background
- Supports Urdu translation
- Tracks learning progress

**Target Users**: Students and professionals learning Physical AI and Humanoid Robotics with diverse backgrounds (software engineers, hardware engineers, researchers).

**Success Metrics**:
- Content quality: pedagogically sound, technically accurate
- RAG accuracy: >85% precision on test queries
- User satisfaction: >80% positive feedback
- Translation quality: maintains educational value in Urdu
- Personalization impact: measurable improvement in learning outcomes

---

**Ready to build an AI-native textbook? Start with the Content Writer agent and work your way through the pipeline!** 🚀
