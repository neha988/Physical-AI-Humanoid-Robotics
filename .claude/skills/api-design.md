---
name: api-design
description: Best practices for RESTful API design including endpoint patterns, validation, error handling, and authentication. Use when designing endpoints, implementing request/response models, handling errors, implementing authentication, structuring APIs, adding rate limiting, or documenting with OpenAPI.
---

# API Design Skill

## Overview
Best practices for designing RESTful APIs for the Physical AI textbook platform. Covers endpoint design, request/response patterns, error handling, validation, and documentation.

## Core Principles

### 1. Resource-Oriented Design
APIs should model resources, not actions.

**Good** ✅:
```
GET    /api/v1/chapters/{id}
POST   /api/v1/users
PUT    /api/v1/users/{id}/profile
GET    /api/v1/rag/query
```

**Bad** ❌:
```
GET    /api/v1/getChapter?id=1
POST   /api/v1/createUser
POST   /api/v1/updateUserProfile
GET    /api/v1/doRagQuery
```

### 2. Consistent Naming Conventions

- **Plural nouns** for collections: `/users`, `/chapters`
- **Lowercase with hyphens**: `/user-profiles`, not `/userProfiles` or `/user_profiles`
- **Nested resources**: `/chapters/{id}/sections`
- **Actions as subcollections**: `/users/{id}/activate` (sparingly)

### 3. HTTP Methods Semantically

| Method | Purpose | Idempotent | Safe |
|--------|---------|------------|------|
| GET | Retrieve resource | Yes | Yes |
| POST | Create resource | No | No |
| PUT | Update/replace resource | Yes | No |
| PATCH | Partial update | No | No |
| DELETE | Remove resource | Yes | No |

## Endpoint Patterns

### Collection and Resource Endpoints

```python
# FastAPI example

from fastapi import APIRouter, HTTPException, status
from typing import List
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/chapters", tags=["Chapters"])

# GET /api/v1/chapters - List all chapters
@router.get("/", response_model=List[ChapterSummary])
async def list_chapters(
    skip: int = 0,
    limit: int = 20,
    difficulty: Optional[str] = None
):
    """
    List all chapters with optional filtering.

    Query Parameters:
    - skip: Number of records to skip (pagination)
    - limit: Max records to return (default: 20, max: 100)
    - difficulty: Filter by difficulty level (beginner|intermediate|advanced)
    """
    if limit > 100:
        raise HTTPException(400, "Limit cannot exceed 100")

    chapters = await chapter_service.get_chapters(
        skip=skip,
        limit=limit,
        difficulty=difficulty
    )

    return chapters

# GET /api/v1/chapters/{chapter_id} - Get specific chapter
@router.get("/{chapter_id}", response_model=ChapterDetail)
async def get_chapter(
    chapter_id: str,
    user = Depends(get_current_user)  # Authentication required
):
    """
    Get detailed chapter content.

    Returns personalized content based on user profile.
    """
    chapter = await chapter_service.get_chapter(chapter_id)

    if not chapter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chapter {chapter_id} not found"
        )

    # Personalize content
    personalized = await personalization_service.adapt(chapter, user.profile)

    return personalized

# POST /api/v1/chapters/{chapter_id}/progress - Mark progress
@router.post("/{chapter_id}/progress", status_code=status.HTTP_204_NO_CONTENT)
async def mark_progress(
    chapter_id: str,
    progress: ChapterProgress,
    user = Depends(get_current_user)
):
    """
    Update user's chapter progress.

    Body:
    {
      "completed": true,
      "time_spent_minutes": 45,
      "quiz_score": 0.85
    }
    """
    await progress_service.update_progress(
        user_id=user.id,
        chapter_id=chapter_id,
        progress=progress
    )

    return  # 204 No Content
```

### Query Parameters vs Path Parameters

**Path Parameters**: For resource identification
```python
GET /api/v1/chapters/{chapter_id}
GET /api/v1/users/{user_id}/profile
```

**Query Parameters**: For filtering, sorting, pagination
```python
GET /api/v1/chapters?difficulty=beginner&limit=10
GET /api/v1/rag/query?mode=context&chapter=chapter-2
```

## Request/Response Models

### Pydantic Models for Validation

```python
from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional, Literal
from datetime import datetime

# Request Models
class UserCreate(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)

    software_experience: Literal["beginner", "intermediate", "advanced"]
    hardware_experience: Literal["beginner", "intermediate", "advanced"]
    math_background: Literal["basic", "intermediate", "advanced"]
    learning_goals: List[str] = Field(..., min_items=1, max_items=10)
    preferred_style: Literal["visual", "code", "theory"]

    @validator('password')
    def password_strength(cls, v):
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

    class Config:
        schema_extra = {
            "example": {
                "email": "student@example.com",
                "password": "SecurePass123!",
                "software_experience": "intermediate",
                "hardware_experience": "beginner",
                "math_background": "intermediate",
                "learning_goals": ["build-robot", "learn-rl"],
                "preferred_style": "code"
            }
        }

class RAGQuery(BaseModel):
    """RAG query request."""
    query: str = Field(..., min_length=1, max_length=1000, description="User's question")
    mode: Literal["context", "full-book"] = Field(
        default="context",
        description="Search within current chapter or entire book"
    )
    chapter_context: Optional[str] = Field(
        None,
        description="Current chapter ID for context mode"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")

# Response Models
class ChapterSummary(BaseModel):
    """Brief chapter information for lists."""
    id: str
    title: str
    chapter_number: int
    difficulty: str
    estimated_minutes: int
    is_completed: bool  # Personalized based on user

class ChapterDetail(BaseModel):
    """Full chapter content."""
    id: str
    title: str
    chapter_number: int
    content: str  # Personalized markdown content
    learning_objectives: List[str]
    prerequisites: List[str]
    related_topics: List[str]
    difficulty_indicator: str  # "easy", "appropriate", "challenging"
    personalization_note: Optional[str]

class RAGResponse(BaseModel):
    """RAG query response."""
    answer: str
    sources: List[Source]
    confidence: float = Field(..., ge=0.0, le=1.0)
    related_topics: List[str]

class Source(BaseModel):
    """Source citation for RAG answer."""
    chapter: str
    section: str
    content_snippet: str
    relevance_score: float

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### Response Envelope Pattern

**Simple Success** (No envelope for simple resources):
```json
{
  "id": "chapter-1",
  "title": "Introduction to Physical AI"
}
```

**List Responses** (Include pagination metadata):
```json
{
  "items": [...],
  "total": 42,
  "skip": 0,
  "limit": 20,
  "has_more": true
}
```

**Complex Responses** (Envelope when needed):
```json
{
  "data": {...},
  "meta": {
    "personalized": true,
    "user_level": "intermediate",
    "cached": false
  }
}
```

## Error Handling

### HTTP Status Codes

```python
from fastapi import HTTPException, status

# 2xx Success
# 200 OK - Request succeeded
# 201 Created - Resource created (return resource in body)
# 204 No Content - Success but no body (e.g., DELETE)

# 4xx Client Errors
# 400 Bad Request - Invalid input
if not is_valid_email(email):
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid email format"
    )

# 401 Unauthorized - Authentication required
if not token:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"}
    )

# 403 Forbidden - Authenticated but no permission
if user.role != "admin":
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin access required"
    )

# 404 Not Found - Resource doesn't exist
chapter = await db.get_chapter(chapter_id)
if not chapter:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Chapter '{chapter_id}' not found"
    )

# 409 Conflict - Request conflicts with current state
if await db.user_exists(email):
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Email already registered"
    )

# 422 Unprocessable Entity - Validation failed (FastAPI auto)
# Pydantic validation errors automatically return 422

# 429 Too Many Requests - Rate limit exceeded
raise HTTPException(
    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    detail="Rate limit exceeded. Try again in 60 seconds.",
    headers={"Retry-After": "60"}
)

# 5xx Server Errors
# 500 Internal Server Error - Unexpected server error
# 503 Service Unavailable - Temporary outage
```

### Structured Error Responses

```python
# Custom exception handler
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime

app = FastAPI()

class AppException(Exception):
    """Base exception with error code."""
    def __init__(self, detail: str, error_code: str):
        self.detail = detail
        self.error_code = error_code

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=400,
        content={
            "detail": exc.detail,
            "error_code": exc.error_code,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )

# Usage
if not rag_result:
    raise AppException(
        detail="No relevant content found for query",
        error_code="RAG_NO_RESULTS"
    )
```

### Error Response Format

```json
{
  "detail": "Chapter 'chapter-99' not found",
  "error_code": "CHAPTER_NOT_FOUND",
  "timestamp": "2025-01-15T10:30:00Z",
  "path": "/api/v1/chapters/chapter-99"
}
```

## Authentication & Authorization

### JWT Bearer Token Pattern

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Verify JWT token and return user.

    Usage:
        @router.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"user_id": user.id}
    """
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user

# Optional authentication (user can be None)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[User]:
    if not credentials:
        return None
    return await get_current_user(credentials)
```

## Pagination

### Offset-Based Pagination

```python
from pydantic import BaseModel
from typing import Generic, TypeVar, List

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    items: List[T]
    total: int
    skip: int
    limit: int
    has_more: bool

@router.get("/chapters", response_model=PaginatedResponse[ChapterSummary])
async def list_chapters(
    skip: int = 0,
    limit: int = 20
):
    chapters = await db.get_chapters(skip=skip, limit=limit)
    total = await db.count_chapters()

    return PaginatedResponse(
        items=chapters,
        total=total,
        skip=skip,
        limit=limit,
        has_more=(skip + limit) < total
    )
```

## Rate Limiting

### Simple Rate Limiting Middleware

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from time import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute=60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP or user ID from token)
        client_id = request.client.host

        now = time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]

        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"}
            )

        # Record this request
        self.requests[client_id].append(now)

        response = await call_next(request)
        return response

# Add to app
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
```

## CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Development
        "https://yourusername.github.io"  # Production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Page-Count"]
)
```

## API Documentation

### OpenAPI/Swagger Integration

```python
from fastapi import FastAPI

app = FastAPI(
    title="Physical AI Textbook API",
    description="RESTful API for AI-native interactive textbook",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_tags=[
        {
            "name": "authentication",
            "description": "User authentication and authorization"
        },
        {
            "name": "chapters",
            "description": "Textbook chapter content"
        },
        {
            "name": "rag",
            "description": "RAG-powered chatbot queries"
        },
        {
            "name": "translation",
            "description": "Content translation services"
        }
    ]
)

# Endpoints automatically documented
@router.post(
    "/query",
    response_model=RAGResponse,
    summary="Query RAG system",
    description="Submit a question to the RAG-powered chatbot and receive an AI-generated answer with source citations.",
    responses={
        200: {"description": "Successful query with answer and sources"},
        401: {"description": "Authentication required"},
        429: {"description": "Rate limit exceeded"}
    }
)
async def query_rag(...):
    ...
```

## Versioning

### URL Path Versioning
```
/api/v1/chapters
/api/v2/chapters  # Future version
```

### Header Versioning (alternative)
```python
from fastapi import Header

@router.get("/chapters")
async def get_chapters(api_version: str = Header(default="v1", alias="X-API-Version")):
    if api_version == "v1":
        return v1_logic()
    elif api_version == "v2":
        return v2_logic()
```

## Best Practices Summary

1. **Use HTTP methods semantically**: GET (read), POST (create), PUT/PATCH (update), DELETE (remove)
2. **Return appropriate status codes**: 2xx success, 4xx client errors, 5xx server errors
3. **Validate all inputs**: Use Pydantic models for automatic validation
4. **Handle errors gracefully**: Structured error responses with codes
5. **Authenticate protected endpoints**: JWT bearer tokens
6. **Implement rate limiting**: Prevent abuse
7. **Version your API**: Plan for future changes
8. **Document thoroughly**: OpenAPI/Swagger auto-generation
9. **Paginate large responses**: Don't return unbounded lists
10. **Use consistent naming**: Plural nouns, lowercase with hyphens

## Anti-Patterns to Avoid

❌ **Verbs in URLs**: `/api/createUser` → Use `/api/users` with POST
❌ **Inconsistent naming**: `/api/user-profile` and `/api/userSettings` → Pick one style
❌ **Ignoring HTTP methods**: Everything as POST → Use correct methods
❌ **Generic error messages**: "Error occurred" → Be specific
❌ **No authentication**: Sensitive endpoints unprotected → Always authenticate
❌ **Exposing internal errors**: Stack traces to users → Sanitize error messages
❌ **No rate limiting**: API can be abused → Add rate limits
❌ **Undocumented APIs**: No docs → Generate OpenAPI documentation
❌ **Breaking changes without versioning**: Existing clients break → Version your API

