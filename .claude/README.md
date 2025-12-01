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
│   ├── qa-specialist.md         # Testing & quality assurance
│   ├── chunking-strategist.md   # Content segmentation for RAG
│   ├── data-pipeline-engineer.md # ETL pipeline implementation
│   ├── deployment-specialist.md # CI/CD and deployment
│   ├── docusaurus-architect.md  # Docusaurus structure & configuration
│   ├── performance-optimizer.md # Performance analysis & optimization
│   └── security-auditor.md      # Security review & auditing
│
├── skills/                      # Reusable skills
│   ├── technical-writing.md     # Educational content patterns
│   ├── api-design.md            # RESTful API best practices
│   ├── vector-embedding.md      # Embedding generation & optimization
│   ├── smart-chunking.md        # Content chunking strategies
│   ├── multilingual-translation.md # Translation with context preservation
│   ├── content-adaptation.md    # Personalization algorithms
│   ├── automation.md            # CI/CD and workflow automation
│   ├── caching-strategies.md    # Caching patterns & optimization
│   ├── compliance.md            # Security compliance & standards
│   ├── data-validation.md       # Input validation & sanitization
│   ├── docker.md                # Container management
│   ├── docusaurus-config.md     # Docusaurus configuration
│   ├── etl-pipeline.md          # Data pipeline design
│   ├── github-actions.md        # GitHub Actions workflows
│   ├── lazy-loading.md          # Performance optimization techniques
│   ├── mdx-authoring.md         # MDX content creation
│   ├── penetration-testing.md   # Security testing
│   ├── performance-analysis.md  # Performance profiling
│   ├── rag-optimization.md      # RAG system optimization
│   ├── react-components.md      # React component development
│   ├── security-review.md       # Code security review
│   ├── semantic-analysis.md     # Content semantic analysis
│   ├── serverless-deployment.md # Serverless deployment strategies
│   └── text-processing.md       # Text processing & chunking
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

### [Chunking Strategist Agent](agents/chunking-strategist.md)
**Purpose**: Design optimal content segmentation strategies for RAG systems.

**Key Responsibilities**:
- Design chunking strategy for technical robotics content
- Balance chunk size vs context preservation
- Implement overlap strategy for continuity
- Handle code blocks, diagrams, and mixed content
- Create metadata enrichment pipeline
- Optimize for retrieval accuracy

**When to Use**:
- "Design chunking strategy for technical content"
- "Optimize content segmentation for RAG"
- "Handle code blocks in chunking"

**Skills**: text-processing, semantic-analysis, rag-optimization

---

### [Data Pipeline Engineer Agent](agents/data-pipeline-engineer.md)
**Purpose**: Build ETL pipeline from Docusaurus content to Qdrant vector database.

**Key Responsibilities**:
- Extract markdown content from Docusaurus
- Transform content with smart chunking
- Generate embeddings and upload to Qdrant
- Automate pipeline on content updates
- Validate data quality and handle errors
- Implement idempotent operations

**When to Use**:
- "Build ETL pipeline for content ingestion"
- "Automate embedding generation and upload"
- "Create data validation pipeline"

**Skills**: etl-pipeline, data-validation, automation

---

### [Deployment Specialist Agent](agents/deployment-specialist.md)
**Purpose**: CI/CD pipeline setup, environment management, and production deployment.

**Key Responsibilities**:
- Set up GitHub Actions workflows
- Configure GitHub Pages deployment for frontend
- Deploy backend to serverless platforms (Vercel/Railway)
- Manage environment variables and secrets
- Implement deployment best practices
- Configure HTTPS and domain settings

**When to Use**:
- "Set up CI/CD pipeline"
- "Deploy to production"
- "Configure GitHub Pages deployment"
- "Set up environment management"

**Skills**: github-actions, docker, serverless-deployment

---

### [Docusaurus Architect Agent](agents/docusaurus-architect.md)
**Purpose**: Expert in Docusaurus site structure, theming, plugins, and configuration.

**Key Responsibilities**:
- Design optimal site structure for technical textbook
- Configure plugins (search, analytics, components)
- Implement custom React components integration
- Set up responsive theme with professional styling
- Optimize build performance
- Configure navigation and sidebar

**When to Use**:
- "Design Docusaurus site structure"
- "Configure Docusaurus plugins"
- "Optimize Docusaurus build performance"
- "Customize Docusaurus theme"

**Skills**: docusaurus-config, react-components, mdx-authoring

---

### [Performance Optimizer Agent](agents/performance-optimizer.md)
**Purpose**: Optimize page load, API response times, and user experience.

**Key Responsibilities**:
- Optimize frontend performance (Lighthouse score >90)
- Improve backend API response times (<3s p95)
- Implement caching strategies (Redis, database-backed)
- Optimize database queries and connections
- Set up performance monitoring and alerts
- Implement lazy loading and code splitting

**When to Use**:
- "Optimize page load speed"
- "Improve API performance"
- "Implement caching strategy"
- "Set up performance monitoring"

**Skills**: performance-analysis, caching-strategies, lazy-loading

---

### [Security Auditor Agent](agents/security-auditor.md)
**Purpose**: Security review for authentication, API endpoints, and data handling.

**Key Responsibilities**:
- Audit API security (input validation, rate limiting, CORS)
- Review authentication implementation (Better-auth)
- Ensure data protection (encryption, secure storage)
- Check dependency security (npm/pip audit)
- Conduct threat modeling (OWASP Top 10)
- Provide security remediation recommendations

**When to Use**:
- "Audit application security"
- "Review authentication security"
- "Check for OWASP vulnerabilities"
- "Validate API security"

**Skills**: security-review, penetration-testing, compliance

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

### [RAG Optimization](skills/rag-optimization.md)
Optimize RAG system performance and accuracy.

**Key Concepts**:
- Fine-tune chunking strategies for better retrieval
- Evaluate and improve embedding models
- Implement re-ranking algorithms
- Query expansion and reformulation
- Monitor performance and identify bottlenecks

**Usage Example**:
```python
# Evaluate chunking strategy
evaluator = RAGEvaluator(test_queries, ground_truth)
metrics = evaluator.evaluate_chunking(chunk_size=400, overlap=50)
print(f"Precision: {metrics.precision}, Recall: {metrics.recall}")
```

---

### [Text Processing](skills/text-processing.md)
Text processing and content manipulation for RAG systems.

**Key Concepts**:
- Parse markdown and extract structure
- Handle code blocks and special formatting
- Extract metadata from frontmatter
- Text normalization and cleaning
- Token counting and management

**Usage Example**:
```python
processor = TextProcessor()
parsed = processor.parse_markdown(content)
# Returns: headers, code_blocks, text_sections, metadata
```

---

### [Semantic Analysis](skills/semantic-analysis.md)
Analyze content semantics for intelligent chunking.

**Key Concepts**:
- Identify semantic boundaries in text
- Extract key concepts and entities
- Determine section importance
- Analyze concept relationships
- Generate semantic metadata

**Usage Example**:
```python
analyzer = SemanticAnalyzer()
boundaries = analyzer.detect_boundaries(content)
concepts = analyzer.extract_concepts(section)
```

---

### [React Components](skills/react-components.md)
React component development for Docusaurus.

**Key Concepts**:
- Create custom React components
- Follow Docusaurus component patterns
- Props and state management
- Component composition
- Integration with MDX

**Usage Example**:
```jsx
// src/components/ChatWidget.js
export default function ChatWidget({ initialOpen = false }) {
  const [isOpen, setIsOpen] = useState(initialOpen);
  return (
    <div className="chat-widget">
      {/* Component implementation */}
    </div>
  );
}
```

---

### [MDX Authoring](skills/mdx-authoring.md)
Create and structure MDX content for Docusaurus.

**Key Concepts**:
- MDX syntax and component embedding
- Frontmatter configuration
- Custom component usage
- Interactive code examples
- Content organization

**Usage Example**:
```mdx
---
title: Forward Kinematics
sidebar_position: 2
---

import CodeDemo from '@site/src/components/CodeDemo';

## Introduction
<CodeDemo language="python" live>
  # Interactive code example
</CodeDemo>
```

---

### [Docusaurus Config](skills/docusaurus-config.md)
Configure and optimize Docusaurus sites.

**Key Concepts**:
- Site configuration (docusaurus.config.js)
- Plugin configuration
- Theme customization
- Navigation and sidebar setup
- Build optimization

**Usage Example**:
```javascript
module.exports = {
  title: 'Physical AI Textbook',
  plugins: [
    ['@docusaurus/plugin-content-docs', {
      sidebarPath: require.resolve('./sidebars.js'),
    }]
  ]
};
```

---

### [Performance Analysis](skills/performance-analysis.md)
Profile and analyze application performance.

**Key Concepts**:
- Lighthouse auditing
- API response time profiling
- Database query analysis
- Bundle size optimization
- Performance metrics tracking

**Usage Example**:
```python
# Profile API endpoint
@profiler.track("chat_endpoint")
async def chat_query(query: str):
    # Function implementation
    pass
```

---

### [Caching Strategies](skills/caching-strategies.md)
Implement efficient caching patterns.

**Key Concepts**:
- Cache-aside pattern
- Redis integration
- Database-backed caching
- Cache invalidation strategies
- TTL management

**Usage Example**:
```python
@cache(ttl=3600, key_prefix="translation")
async def translate_content(text: str, target_lang: str):
    return await translator.translate(text, target_lang)
```

---

### [Lazy Loading](skills/lazy-loading.md)
Optimize resource loading for performance.

**Key Concepts**:
- Component lazy loading
- Image lazy loading
- Route-based code splitting
- Dynamic imports
- Intersection Observer

**Usage Example**:
```jsx
const ChatWidget = lazy(() => import('./components/ChatWidget'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <ChatWidget />
    </Suspense>
  );
}
```

---

### [Security Review](skills/security-review.md)
Code security analysis and vulnerability assessment.

**Key Concepts**:
- OWASP Top 10 vulnerabilities
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- Authentication security

**Usage Example**:
```python
# Input validation
from pydantic import BaseModel, validator

class QueryInput(BaseModel):
    query: str

    @validator('query')
    def validate_query(cls, v):
        if len(v) > 1000:
            raise ValueError('Query too long')
        return sanitize_input(v)
```

---

### [Penetration Testing](skills/penetration-testing.md)
Security testing and vulnerability discovery.

**Key Concepts**:
- API endpoint testing
- Authentication bypass testing
- Rate limiting validation
- CORS configuration testing
- Dependency vulnerability scanning

**Usage Example**:
```bash
# Run security audit
npm audit
pip-audit
# Test rate limiting
ab -n 1000 -c 10 https://api.example.com/chat
```

---

### [Compliance](skills/compliance.md)
Ensure security compliance and standards adherence.

**Key Concepts**:
- GDPR compliance
- Data privacy requirements
- API security standards
- Access control policies
- Audit logging

---

### [GitHub Actions](skills/github-actions.md)
Automate workflows with GitHub Actions.

**Key Concepts**:
- CI/CD pipeline setup
- Automated testing
- Deployment workflows
- Environment management
- Secrets handling

**Usage Example**:
```yaml
name: Test and Deploy
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm test
```

---

### [Docker](skills/docker.md)
Container management and optimization.

**Key Concepts**:
- Dockerfile creation
- Image optimization
- Docker Compose
- Multi-stage builds
- Container networking

**Usage Example**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

---

### [Serverless Deployment](skills/serverless-deployment.md)
Deploy applications to serverless platforms.

**Key Concepts**:
- Vercel deployment
- Railway.app deployment
- Environment configuration
- Serverless function optimization
- Cold start mitigation

---

### [ETL Pipeline](skills/etl-pipeline.md)
Design and implement data pipelines.

**Key Concepts**:
- Extract from multiple sources
- Transform with validation
- Load to vector database
- Error handling and retry logic
- Idempotent operations

**Usage Example**:
```python
def etl_pipeline():
    # Extract
    content = extract_from_docusaurus()
    # Transform
    chunks = chunk_and_enrich(content)
    # Load
    upload_to_qdrant(chunks)
```

---

### [Data Validation](skills/data-validation.md)
Validate and sanitize input data.

**Key Concepts**:
- Pydantic models
- Input sanitization
- Type validation
- Business rule validation
- Error handling

**Usage Example**:
```python
from pydantic import BaseModel, Field

class ChunkData(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    chapter: str
    metadata: dict
```

---

### [Automation](skills/automation.md)
Automate repetitive tasks and workflows.

**Key Concepts**:
- Script automation
- Scheduled jobs
- Event-driven automation
- CI/CD integration
- Monitoring and alerting

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
| Chunking Strategist | text-processing, semantic-analysis | rag-optimization |
| Data Pipeline Engineer | etl-pipeline, data-validation | automation |
| Deployment Specialist | github-actions, docker | serverless-deployment |
| Docusaurus Architect | docusaurus-config, react-components | mdx-authoring |
| Performance Optimizer | performance-analysis, caching-strategies | lazy-loading |
| Security Auditor | security-review, penetration-testing | compliance |

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
