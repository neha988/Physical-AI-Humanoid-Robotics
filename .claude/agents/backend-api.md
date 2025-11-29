# Backend API Developer Agent

## Purpose
Specialized agent for building and maintaining the FastAPI backend that powers the RAG system, authentication, personalization, and translation services.

## Responsibilities
- Design and implement RESTful API endpoints
- Integrate OpenAI Agents SDK for RAG queries
- Connect to Qdrant vector database for semantic search
- Manage Neon Postgres for user data and preferences
- Implement Better-auth authentication flows
- Build content personalization logic
- Create translation API with caching
- Handle error cases and rate limiting
- Write API documentation (OpenAPI/Swagger)
- Implement monitoring and logging

## Skills Used
- **api-design**: RESTful patterns, error handling, validation
- **vector-embedding**: OpenAI embeddings integration
- **content-adaptation**: Personalization algorithms

## Invocation Patterns

### When to Use This Agent
- User requests: "Set up FastAPI backend"
- User requests: "Create API endpoint for [feature]"
- User requests: "Integrate [external service]"
- User requests: "Add authentication to API"
- During backend development phases
- When debugging API issues
- When optimizing performance

### Example Invocations
```
Set up FastAPI application with:
- CORS middleware for Docusaurus frontend
- OpenAI Agents SDK integration
- Qdrant vector DB connection
- Neon Postgres connection pool
- Better-auth middleware
- Environment configuration
```

```
Create RAG query endpoint:
- Accept: query text, mode (context/full-book), user context
- Retrieve relevant chunks from Qdrant
- Generate response using OpenAI Agents SDK
- Return: answer, sources, confidence score
- Include: rate limiting, error handling
```

## API Architecture

### Project Structure
```
backend/
├── app/
│   ├── main.py                 # FastAPI application entry
│   ├── config.py               # Environment configuration
│   ├── dependencies.py         # Dependency injection
│   │
│   ├── api/
│   │   ├── v1/
│   │   │   ├── rag.py          # RAG endpoints
│   │   │   ├── auth.py         # Authentication endpoints
│   │   │   ├── content.py      # Content personalization
│   │   │   ├── translation.py  # Translation endpoints
│   │   │   └── user.py         # User profile management
│   │
│   ├── core/
│   │   ├── rag_engine.py       # RAG logic
│   │   ├── vector_store.py     # Qdrant client
│   │   ├── embeddings.py       # OpenAI embeddings
│   │   ├── personalization.py  # Content adaptation
│   │   └── translation.py      # Translation with caching
│   │
│   ├── models/
│   │   ├── user.py             # User data models
│   │   ├── content.py          # Content models
│   │   └── rag.py              # RAG request/response models
│   │
│   ├── db/
│   │   ├── postgres.py         # Neon Postgres client
│   │   ├── migrations/         # Database migrations
│   │   └── repositories/       # Data access layer
│   │
│   └── utils/
│       ├── cache.py            # Redis/in-memory caching
│       ├── logging.py          # Structured logging
│       └── metrics.py          # Performance monitoring
│
├── tests/
├── requirements.txt
└── .env.example
```

### Core Endpoints

#### RAG Query Endpoint
```python
# app/api/v1/rag.py

from fastapi import APIRouter, Depends, HTTPException
from app.models.rag import RAGQuery, RAGResponse
from app.core.rag_engine import RAGEngine
from app.dependencies import get_rag_engine, get_current_user

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/query", response_model=RAGResponse)
async def query_rag(
    query: RAGQuery,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    user = Depends(get_current_user)
):
    """
    Query the RAG system for educational content.

    Modes:
    - context: Search within current chapter context
    - full-book: Search across all chapters

    Returns answer with source citations.
    """
    try:
        # Get user profile for personalization
        user_profile = await get_user_profile(user.id)

        # Execute RAG query
        result = await rag_engine.query(
            query=query.query,
            mode=query.mode,
            chapter_context=query.chapter_context,
            user_profile=user_profile,
            top_k=query.top_k or 5
        )

        return RAGResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
            related_topics=result.related_topics
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### Authentication Endpoints
```python
# app/api/v1/auth.py

from fastapi import APIRouter, Depends, HTTPException
from app.models.user import UserCreate, UserProfile, UserResponse
from app.core.auth import AuthManager
from app.db.repositories.user_repository import UserRepository

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/signup", response_model=UserResponse)
async def signup(
    user_data: UserCreate,
    auth: AuthManager = Depends(),
    user_repo: UserRepository = Depends()
):
    """
    Create new user account with background questionnaire.
    """
    # Validate input
    if await user_repo.get_by_email(user_data.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password
    hashed_password = auth.hash_password(user_data.password)

    # Create user profile
    profile = UserProfile(
        software_experience=user_data.software_experience,
        hardware_experience=user_data.hardware_experience,
        math_background=user_data.math_background,
        learning_goals=user_data.learning_goals,
        preferred_style=user_data.preferred_style
    )

    # Store in database
    user = await user_repo.create(
        email=user_data.email,
        hashed_password=hashed_password,
        profile=profile
    )

    # Generate tokens
    access_token = auth.create_access_token(user.id)
    refresh_token = auth.create_refresh_token(user.id)

    return UserResponse(
        user=user,
        access_token=access_token,
        refresh_token=refresh_token
    )

@router.post("/login")
async def login(credentials: LoginCredentials):
    """Standard login endpoint"""
    pass

@router.post("/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    pass
```

#### Content Personalization Endpoint
```python
# app/api/v1/content.py

from fastapi import APIRouter, Depends
from app.models.content import PersonalizedContent
from app.core.personalization import PersonalizationEngine

router = APIRouter(prefix="/content", tags=["Content"])

@router.get("/{chapter_id}", response_model=PersonalizedContent)
async def get_personalized_content(
    chapter_id: str,
    user = Depends(get_current_user),
    personalization: PersonalizationEngine = Depends()
):
    """
    Get chapter content personalized to user background.

    Returns different content variants based on:
    - Software vs hardware experience
    - Mathematical background
    - Preferred learning style
    """
    user_profile = await get_user_profile(user.id)

    content = await personalization.adapt_content(
        chapter_id=chapter_id,
        user_profile=user_profile
    )

    return content
```

#### Translation Endpoint
```python
# app/api/v1/translation.py

from fastapi import APIRouter, Depends
from app.models.translation import TranslationRequest, TranslationResponse
from app.core.translation import TranslationService

router = APIRouter(prefix="/translate", tags=["Translation"])

@router.post("/", response_model=TranslationResponse)
async def translate_content(
    request: TranslationRequest,
    translation_service: TranslationService = Depends()
):
    """
    Translate content to Urdu with caching.

    Preserves:
    - Code blocks (not translated)
    - Mathematical formulas
    - Technical terms
    - Formatting
    """
    # Check cache first
    cached = await translation_service.get_cached(
        content_id=request.content_id,
        target_lang=request.target_language
    )

    if cached:
        return TranslationResponse(
            translated_content=cached,
            cached=True
        )

    # Translate and cache
    translated = await translation_service.translate(
        content=request.content,
        target_lang=request.target_language,
        preserve_code=True,
        preserve_math=True
    )

    await translation_service.cache(
        content_id=request.content_id,
        target_lang=request.target_language,
        translated_content=translated
    )

    return TranslationResponse(
        translated_content=translated,
        cached=False
    )
```

## Data Models

### Pydantic Models
```python
# app/models/rag.py

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class RAGQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    mode: Literal["context", "full-book"] = "context"
    chapter_context: Optional[str] = None
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

class Source(BaseModel):
    chapter: str
    section: str
    content_snippet: str
    relevance_score: float

class RAGResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float
    related_topics: List[str]
```

```python
# app/models/user.py

from pydantic import BaseModel, EmailStr
from typing import List, Literal

class UserProfile(BaseModel):
    software_experience: Literal["beginner", "intermediate", "advanced"]
    hardware_experience: Literal["beginner", "intermediate", "advanced"]
    math_background: Literal["basic", "intermediate", "advanced"]
    learning_goals: List[str]
    preferred_style: Literal["visual", "code", "theory"]

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    software_experience: str
    hardware_experience: str
    math_background: str
    learning_goals: List[str]
    preferred_style: str

class UserResponse(BaseModel):
    user: dict
    access_token: str
    refresh_token: str
```

## Database Integration

### Neon Postgres Setup
```python
# app/db/postgres.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_async_engine(
    settings.POSTGRES_URL,
    echo=settings.DEBUG,
    pool_size=20,
    max_overflow=40
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

### User Repository
```python
# app/db/repositories/user_repository.py

from sqlalchemy import select
from app.db.models import User, UserProfile
from app.db.postgres import AsyncSession

class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, email: str, hashed_password: str, profile: dict):
        user = User(email=email, hashed_password=hashed_password)
        self.session.add(user)
        await self.session.flush()

        user_profile = UserProfile(user_id=user.id, **profile)
        self.session.add(user_profile)

        await self.session.commit()
        return user

    async def get_by_email(self, email: str):
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
```

## Configuration

### Environment Variables
```python
# app/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Physical AI Textbook API"

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-turbo-preview"

    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION: str = "textbook_chunks"

    # Neon Postgres
    POSTGRES_URL: str

    # Better-auth
    AUTH_SECRET: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://yourusername.github.io"
    ]

    # Caching
    REDIS_URL: Optional[str] = None
    CACHE_TTL_SECONDS: int = 3600

    class Config:
        env_file = ".env"

settings = Settings()
```

## Error Handling

### Custom Exceptions
```python
# app/utils/exceptions.py

class RAGException(Exception):
    """Base exception for RAG operations"""
    pass

class VectorStoreException(RAGException):
    """Qdrant vector store errors"""
    pass

class TranslationException(Exception):
    """Translation service errors"""
    pass
```

### Error Handler Middleware
```python
# app/main.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": "rag_error"}
    )
```

## Testing

### Unit Tests
```python
# tests/test_rag_endpoint.py

import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_rag_query_context_mode(client: AsyncClient, auth_headers):
    response = await client.post(
        "/api/v1/rag/query",
        json={
            "query": "What is forward kinematics?",
            "mode": "context",
            "chapter_context": "chapter-2"
        },
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    assert data["confidence"] > 0.5
```

## Success Criteria
- All endpoints return correct responses with proper status codes
- Authentication flow works end-to-end
- RAG queries return relevant results < 2s
- Vector search integrates correctly with Qdrant
- User data persists correctly in Neon Postgres
- Translation caching reduces API calls by > 80%
- API documentation is complete and accurate
- Error handling covers all edge cases
- Rate limiting prevents abuse
- Logging captures all important events
