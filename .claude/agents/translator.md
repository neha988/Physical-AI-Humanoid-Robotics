# Translation Service Agent

## Purpose
Specialized agent for implementing Urdu translation services with intelligent caching, context preservation, and technical term handling for educational content.

## Responsibilities
- Implement translation API using OpenAI GPT-4
- Build translation caching system (Redis or database)
- Preserve code blocks, formulas, and technical terms
- Handle RTL (right-to-left) text formatting
- Maintain context and educational tone in translations
- Implement fallback mechanisms for translation errors
- Optimize translation costs through batching and caching
- Create translation quality checks
- Handle mixed language content (English technical terms in Urdu text)

## Skills Used
- **multilingual-translation**: Translation with context preservation
- **api-design**: Translation endpoint patterns
- **content-adaptation**: Maintaining pedagogical quality across languages

## Invocation Patterns

### When to Use This Agent
- User requests: "Implement Urdu translation"
- User requests: "Add translation caching"
- User requests: "Translate content while preserving code"
- During translation feature development
- When optimizing translation costs
- When debugging translation quality issues

### Example Invocations
```
Implement translation service:
- Source: English educational content
- Target: Urdu
- Preserve: Code blocks, formulas, technical terms
- Cache: Translated content in Postgres
- API: OpenAI GPT-4 with custom prompts
```

```
Translate chapter with context:
- Chapter: "Forward Kinematics"
- Preserve: Mathematical notation, variable names, code
- Maintain: Educational tone, clarity
- Cache result for future requests
```

## Translation Architecture

### System Flow

```
┌─────────────────────────────────────────────────┐
│         Translation Request                      │
│  (content_id, target_language='ur')             │
└─────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │   Check Cache       │
        │  (Postgres table)   │
        └─────────────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    Cache Hit             Cache Miss
         │                     │
         ▼                     ▼
    ┌─────────┐      ┌──────────────────┐
    │ Return  │      │  Content Parsing │
    │ Cached  │      │  • Split sections │
    └─────────┘      │  • Identify code  │
                     │  • Extract formulas│
                     └──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Translation     │
                     │  (OpenAI GPT-4)  │
                     │  • Custom prompt │
                     │  • Batch requests│
                     └──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Post-Processing │
                     │  • Restore code   │
                     │  • Verify quality │
                     │  • Format RTL     │
                     └──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Cache Result    │
                     │  (Save to DB)    │
                     └──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Return          │
                     │  Translation     │
                     └──────────────────┘
```

## Implementation

### Translation Service Core
```python
# app/core/translation.py

from typing import Dict, List, Optional
from openai import AsyncOpenAI
import re
from app.db.repositories.translation_repository import TranslationRepository

class TranslationService:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        translation_repo: TranslationRepository
    ):
        self.openai = openai_client
        self.translation_repo = translation_repo
        self.model = "gpt-4-turbo-preview"

        # Technical terms that should NOT be translated
        self.technical_terms = [
            "forward kinematics", "inverse kinematics", "jacobian",
            "neural network", "reinforcement learning", "policy",
            "trajectory", "end-effector", "actuator", "sensor",
            "RGB", "depth", "point cloud", "ROS", "API", "SDK"
        ]

    async def translate(
        self,
        content: str,
        content_id: str,
        target_language: str = "ur",
        preserve_code: bool = True,
        preserve_math: bool = True
    ) -> str:
        """
        Translate content to target language with preservation rules.

        Args:
            content: Source content (markdown)
            content_id: Unique identifier for caching
            target_language: Target language code (ur for Urdu)
            preserve_code: Don't translate code blocks
            preserve_math: Don't translate math formulas

        Returns:
            Translated content
        """
        # 1. Check cache
        cached = await self.get_cached(content_id, target_language)
        if cached:
            return cached

        # 2. Parse and extract preservable elements
        parsed = self._parse_content(content, preserve_code, preserve_math)

        # 3. Translate text portions
        translated_text = await self._translate_text(
            text=parsed["text"],
            target_language=target_language,
            context=parsed.get("context", "")
        )

        # 4. Reassemble content
        final_content = self._reassemble_content(
            translated_text=translated_text,
            preserved_elements=parsed["preserved_elements"]
        )

        # 5. Cache result
        await self.cache(content_id, target_language, final_content)

        return final_content

    def _parse_content(
        self,
        content: str,
        preserve_code: bool,
        preserve_math: bool
    ) -> Dict:
        """
        Parse content and extract elements to preserve.

        Returns dict with:
        - text: Translatable text with placeholders
        - preserved_elements: List of (placeholder, original) tuples
        - context: Educational context for better translation
        """
        preserved_elements = []
        text = content
        placeholder_counter = 0

        # Extract code blocks
        if preserve_code:
            code_pattern = r'```[\s\S]*?```'
            for match in re.finditer(code_pattern, text):
                placeholder = f"__CODE_BLOCK_{placeholder_counter}__"
                preserved_elements.append((placeholder, match.group(0)))
                text = text.replace(match.group(0), placeholder, 1)
                placeholder_counter += 1

            # Inline code
            inline_code_pattern = r'`[^`]+`'
            for match in re.finditer(inline_code_pattern, text):
                placeholder = f"__INLINE_CODE_{placeholder_counter}__"
                preserved_elements.append((placeholder, match.group(0)))
                text = text.replace(match.group(0), placeholder, 1)
                placeholder_counter += 1

        # Extract math formulas
        if preserve_math:
            # LaTeX math blocks
            math_block_pattern = r'\$\$[\s\S]*?\$\$'
            for match in re.finditer(math_block_pattern, text):
                placeholder = f"__MATH_BLOCK_{placeholder_counter}__"
                preserved_elements.append((placeholder, match.group(0)))
                text = text.replace(match.group(0), placeholder, 1)
                placeholder_counter += 1

            # Inline math
            inline_math_pattern = r'\$[^$]+\$'
            for match in re.finditer(inline_math_pattern, text):
                placeholder = f"__INLINE_MATH_{placeholder_counter}__"
                preserved_elements.append((placeholder, match.group(0)))
                text = text.replace(match.group(0), placeholder, 1)
                placeholder_counter += 1

        # Protect technical terms
        for term in self.technical_terms:
            if term.lower() in text.lower():
                placeholder = f"__TERM_{placeholder_counter}__"
                preserved_elements.append((placeholder, term))
                text = re.sub(
                    re.escape(term),
                    placeholder,
                    text,
                    flags=re.IGNORECASE,
                    count=1
                )
                placeholder_counter += 1

        return {
            "text": text,
            "preserved_elements": preserved_elements,
            "context": "educational robotics textbook"
        }

    async def _translate_text(
        self,
        text: str,
        target_language: str,
        context: str = ""
    ) -> str:
        """
        Translate text using OpenAI GPT-4 with educational context.
        """
        system_prompt = self._build_translation_prompt(target_language, context)

        response = await self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,  # Lower temperature for consistency
            max_tokens=4000
        )

        return response.choices[0].message.content

    def _build_translation_prompt(self, target_language: str, context: str) -> str:
        """Build specialized translation prompt for educational content."""
        if target_language == "ur":
            return f"""You are an expert translator specializing in educational content about Physical AI and Humanoid Robotics.

Your task:
- Translate the following English text to Urdu
- Maintain educational tone and clarity
- Preserve technical accuracy
- Keep placeholder markers (e.g., __CODE_BLOCK_0__) exactly as they are
- Use appropriate Urdu terminology for robotics and AI concepts
- Maintain the same formatting structure (headings, lists, etc.)

Context: {context}

Important guidelines:
1. Technical terms in placeholders should NOT be translated
2. Maintain clear, simple language suitable for learners
3. Preserve the pedagogical structure (explanations → examples → practice)
4. Use formal Urdu appropriate for educational content
5. Keep numbers, variable names, and symbols unchanged

Translate only the text, keeping all placeholders intact."""
        else:
            return f"Translate the following text to {target_language}. Preserve all placeholders."

    def _reassemble_content(
        self,
        translated_text: str,
        preserved_elements: List[tuple]
    ) -> str:
        """
        Reassemble translated text with preserved elements.
        """
        result = translated_text

        # Replace all placeholders with original content
        for placeholder, original in preserved_elements:
            result = result.replace(placeholder, original)

        return result

    async def get_cached(
        self,
        content_id: str,
        target_language: str
    ) -> Optional[str]:
        """Retrieve cached translation from database."""
        return await self.translation_repo.get_translation(
            content_id=content_id,
            target_language=target_language
        )

    async def cache(
        self,
        content_id: str,
        target_language: str,
        translated_content: str
    ):
        """Store translation in cache."""
        await self.translation_repo.save_translation(
            content_id=content_id,
            target_language=target_language,
            translated_content=translated_content
        )

    async def batch_translate(
        self,
        contents: List[Dict[str, str]]
    ) -> List[str]:
        """
        Batch translate multiple content pieces.
        More cost-efficient for large translations.

        Args:
            contents: List of {"content_id": str, "content": str}

        Returns:
            List of translated contents
        """
        translations = []

        for item in contents:
            translated = await self.translate(
                content=item["content"],
                content_id=item["content_id"],
                target_language="ur"
            )
            translations.append(translated)

        return translations
```

### Database Schema for Caching
```python
# app/db/models.py (add to existing models)

from sqlalchemy import Column, String, Text, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID
import uuid

class TranslationCache(Base):
    __tablename__ = "translation_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(String, nullable=False)
    source_language = Column(String, default="en")
    target_language = Column(String, nullable=False)
    translated_content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite index for fast lookups
    __table_args__ = (
        Index('idx_content_lang', 'content_id', 'target_language'),
    )
```

### Translation Repository
```python
# app/db/repositories/translation_repository.py

from sqlalchemy import select
from app.db.models import TranslationCache
from app.db.postgres import AsyncSession

class TranslationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_translation(
        self,
        content_id: str,
        target_language: str
    ) -> Optional[str]:
        """Get cached translation."""
        result = await self.session.execute(
            select(TranslationCache).where(
                TranslationCache.content_id == content_id,
                TranslationCache.target_language == target_language
            )
        )
        cache = result.scalar_one_or_none()
        return cache.translated_content if cache else None

    async def save_translation(
        self,
        content_id: str,
        target_language: str,
        translated_content: str
    ):
        """Save translation to cache."""
        # Check if exists
        existing = await self.get_translation(content_id, target_language)

        if existing:
            # Update
            await self.session.execute(
                update(TranslationCache)
                .where(
                    TranslationCache.content_id == content_id,
                    TranslationCache.target_language == target_language
                )
                .values(
                    translated_content=translated_content,
                    updated_at=datetime.utcnow()
                )
            )
        else:
            # Insert
            cache = TranslationCache(
                content_id=content_id,
                source_language="en",
                target_language=target_language,
                translated_content=translated_content
            )
            self.session.add(cache)

        await self.session.commit()
```

### Translation API Endpoint
```python
# app/api/v1/translation.py (complete version)

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.core.translation import TranslationService
from app.dependencies import get_translation_service

router = APIRouter(prefix="/translate", tags=["Translation"])

class TranslationRequest(BaseModel):
    content_id: str
    content: str
    target_language: str = "ur"
    preserve_code: bool = True
    preserve_math: bool = True

class TranslationResponse(BaseModel):
    content_id: str
    translated_content: str
    target_language: str
    cached: bool

@router.post("/", response_model=TranslationResponse)
async def translate_content(
    request: TranslationRequest,
    translation_service: TranslationService = Depends(get_translation_service)
):
    """
    Translate content to Urdu with intelligent caching.
    """
    # Check cache first
    cached = await translation_service.get_cached(
        content_id=request.content_id,
        target_language=request.target_language
    )

    if cached:
        return TranslationResponse(
            content_id=request.content_id,
            translated_content=cached,
            target_language=request.target_language,
            cached=True
        )

    # Translate
    translated = await translation_service.translate(
        content=request.content,
        content_id=request.content_id,
        target_language=request.target_language,
        preserve_code=request.preserve_code,
        preserve_math=request.preserve_math
    )

    return TranslationResponse(
        content_id=request.content_id,
        translated_content=translated,
        target_language=request.target_language,
        cached=False
    )

@router.post("/batch")
async def batch_translate(
    requests: List[TranslationRequest],
    translation_service: TranslationService = Depends(get_translation_service)
):
    """Batch translate multiple content pieces."""
    contents = [
        {"content_id": req.content_id, "content": req.content}
        for req in requests
    ]

    translations = await translation_service.batch_translate(contents)

    return {
        "translations": [
            {
                "content_id": req.content_id,
                "translated_content": trans
            }
            for req, trans in zip(requests, translations)
        ]
    }
```

## Frontend Integration

### Translation Hook (React)
```typescript
// src/hooks/useTranslation.ts

import { useState, useEffect } from 'react';
import { apiClient } from '../api/client';

export function useTranslation(contentId: string, originalContent: string) {
  const [language, setLanguage] = useState<'en' | 'ur'>('en');
  const [content, setContent] = useState(originalContent);
  const [isLoading, setIsLoading] = useState(false);

  const toggleLanguage = async () => {
    if (language === 'en') {
      // Translate to Urdu
      setIsLoading(true);
      try {
        const response = await apiClient.post('/api/v1/translate', {
          content_id: contentId,
          content: originalContent,
          target_language: 'ur'
        });

        setContent(response.data.translated_content);
        setLanguage('ur');
      } catch (error) {
        console.error('Translation failed:', error);
      } finally {
        setIsLoading(false);
      }
    } else {
      // Back to English
      setContent(originalContent);
      setLanguage('en');
    }
  };

  return {
    language,
    content,
    isLoading,
    toggleLanguage
  };
}
```

## Cost Optimization

### Translation Cost Estimation
```python
# Estimate: ~$0.01 per 1000 tokens (GPT-4)
# Average chapter: ~5000 words = ~6500 tokens
# Cost per chapter translation: ~$0.065
# 4 chapters: ~$0.26

# With caching: only pay once per content update
# Expected savings: > 95% (users request same translations)
```

## Testing

### Translation Tests
```python
# tests/test_translation.py

@pytest.mark.asyncio
async def test_translate_preserves_code():
    content = """
    # Forward Kinematics
    Calculate position:
    ```python
    def forward_kinematics(theta):
        return np.array([x, y])
    ```
    """

    translated = await translation_service.translate(
        content=content,
        content_id="test-1",
        target_language="ur"
    )

    # Code block should be preserved
    assert "```python" in translated
    assert "def forward_kinematics" in translated

@pytest.mark.asyncio
async def test_translation_caching():
    # First request
    result1 = await translation_service.translate(
        content="Hello world",
        content_id="test-2",
        target_language="ur"
    )

    # Second request (should be cached)
    result2 = await translation_service.get_cached("test-2", "ur")

    assert result1 == result2
```

## Success Criteria
- Content translates to Urdu accurately
- Code blocks and formulas preserved correctly
- Technical terms remain in English where appropriate
- Translation caching reduces API calls by > 90%
- RTL formatting works properly
- Educational tone maintained in translations
- Translation latency < 3s for new content
- Cache hit rate > 85% for returning users
- Cost per translation meets budget targets
- Quality checks pass for sample translations
