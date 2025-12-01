---
name: multilingual-translation
description: Patterns for translating educational content while preserving technical accuracy and code integrity, focusing on English to Urdu translation with caching. Use when implementing translation features, preserving code/math in translations, handling RTL text, implementing translation caching, or ensuring pedagogical quality across languages.
---

# Multilingual Translation Skill

## Overview
Patterns and best practices for translating educational content while preserving technical accuracy, code integrity, and pedagogical value. Focuses on English → Urdu translation with caching and context preservation.

## Translation Challenges

### Unique Challenges for Educational Content

1. **Technical Terms**: "forward kinematics", "Jacobian matrix" - keep or translate?
2. **Code Blocks**: Must remain unchanged in any language
3. **Mathematical Notation**: Universal but needs context in Urdu
4. **Pedagogical Tone**: Maintain clarity and teaching quality
5. **Mixed Content**: Code + explanation in same document
6. **RTL (Right-to-Left)**: Urdu layout implications

## Translation Strategy

### What to Translate, What to Preserve

| Content Type | Strategy | Example |
|--------------|----------|---------|
| Technical terms (standard) | **Keep in English** | "forward kinematics", "robot", "API" |
| Common technical terms | **Translate + add English** | "حرکیات (kinematics)" |
| Code | **Never translate** | `def forward_kinematics():` stays |
| Math formulas | **Never translate** | $\theta_1, \theta_2$ stays |
| Variable names | **Never translate** | `theta`, `x`, `y` stay |
| Explanatory text | **Translate** | "Calculate position" → "پوزیشن کا حساب لگائیں" |
| Comments in code | **Translate** | `# Calculate FK` → `# FK کا حساب` |
| Markdown formatting | **Preserve** | `##`, `**bold**` stay |

## Implementation

### Translation Pipeline

```python
from openai import AsyncOpenAI
from typing import Dict, List, Optional
import re

class EducationalTranslator:
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client
        self.model = "gpt-4-turbo-preview"

        # Technical terms to NEVER translate
        self.preserve_terms = {
            # Kinematics
            "forward kinematics", "inverse kinematics", "jacobian",
            "end-effector", "denavit-hartenberg", "dh parameters",

            # AI/ML
            "reinforcement learning", "neural network", "deep learning",
            "policy", "reward", "gradient descent", "backpropagation",

            # Robotics
            "robot", "actuator", "sensor", "encoder", "servo",
            "pid controller", "trajectory", "motion planning",

            # Computer Science
            "api", "sdk", "framework", "library", "function",
            "class", "method", "array", "algorithm",

            # File/tech formats
            "python", "numpy", "pytorch", "ros", "urdf",
            "json", "yaml", "markdown"
        }

    async def translate_content(
        self,
        content: str,
        source_lang: str = "en",
        target_lang: str = "ur",
        preserve_code: bool = True,
        preserve_math: bool = True
    ) -> str:
        """
        Translate educational content with preservation rules.

        Process:
        1. Extract and protect code blocks, formulas, technical terms
        2. Translate remaining text using specialized prompt
        3. Restore protected elements
        4. Validate translation quality

        Args:
            content: Source markdown content
            source_lang: Source language code ("en")
            target_lang: Target language code ("ur")
            preserve_code: Don't translate code blocks
            preserve_math: Don't translate math formulas

        Returns:
            Translated content
        """
        # 1. Parse and protect elements
        parsed = self._parse_content(content, preserve_code, preserve_math)

        # 2. Translate text portions
        translated_text = await self._translate_with_context(
            text=parsed["translatable_text"],
            source_lang=source_lang,
            target_lang=target_lang,
            context=parsed["content_metadata"]
        )

        # 3. Reassemble with preserved elements
        final_content = self._reassemble(
            translated_text=translated_text,
            preserved_elements=parsed["preserved_elements"]
        )

        # 4. Validate (optional but recommended)
        self._validate_translation(content, final_content)

        return final_content

    def _parse_content(
        self,
        content: str,
        preserve_code: bool,
        preserve_math: bool
    ) -> Dict:
        """
        Extract elements that should not be translated.

        Returns:
        {
            "translatable_text": str with placeholders,
            "preserved_elements": {placeholder: original_text},
            "content_metadata": {metadata for context}
        }
        """
        preserved = {}
        placeholder_id = 0
        text = content

        # Extract code blocks
        if preserve_code:
            # Fenced code blocks (```...```)
            def replace_code_block(match):
                nonlocal placeholder_id
                placeholder = f"__CODE_BLOCK_{placeholder_id}__"
                preserved[placeholder] = match.group(0)
                placeholder_id += 1
                return placeholder

            text = re.sub(r'```[\s\S]*?```', replace_code_block, text)

            # Inline code (`...`)
            def replace_inline_code(match):
                nonlocal placeholder_id
                placeholder = f"__CODE_{placeholder_id}__"
                preserved[placeholder] = match.group(0)
                placeholder_id += 1
                return placeholder

            text = re.sub(r'`[^`\n]+`', replace_inline_code, text)

        # Extract math formulas
        if preserve_math:
            # Block math ($$...$$)
            def replace_math_block(match):
                nonlocal placeholder_id
                placeholder = f"__MATH_BLOCK_{placeholder_id}__"
                preserved[placeholder] = match.group(0)
                placeholder_id += 1
                return placeholder

            text = re.sub(r'\$\$[\s\S]*?\$\$', replace_math_block, text)

            # Inline math ($...$)
            def replace_inline_math(match):
                nonlocal placeholder_id
                placeholder = f"__MATH_{placeholder_id}__"
                preserved[placeholder] = match.group(0)
                placeholder_id += 1
                return placeholder

            text = re.sub(r'\$[^$\n]+\$', replace_inline_math, text)

        # Protect technical terms
        for term in self.preserve_terms:
            if term.lower() in text.lower():
                placeholder = f"__TERM_{placeholder_id}__"

                # Case-insensitive replace (preserve original case)
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                match = pattern.search(text)

                if match:
                    preserved[placeholder] = match.group(0)
                    text = pattern.sub(placeholder, text, count=1)
                    placeholder_id += 1

        # Extract content metadata for context
        metadata = {
            "has_code": "```" in content,
            "has_math": "$$" in content or "$" in content,
            "estimated_difficulty": self._estimate_difficulty(content),
            "content_type": self._classify_content_type(content)
        }

        return {
            "translatable_text": text,
            "preserved_elements": preserved,
            "content_metadata": metadata
        }

    async def _translate_with_context(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Dict
    ) -> str:
        """
        Translate text using GPT-4 with specialized educational prompt.
        """
        system_prompt = self._build_translation_prompt(
            source_lang=source_lang,
            target_lang=target_lang,
            context=context
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,  # Lower temp for consistency
            max_tokens=4000
        )

        return response.choices[0].message.content

    def _build_translation_prompt(
        self,
        source_lang: str,
        target_lang: str,
        context: Dict
    ) -> str:
        """
        Build specialized prompt for educational content translation.
        """
        if target_lang == "ur":
            prompt = f"""You are an expert translator specializing in educational content about Physical AI, Robotics, and Machine Learning.

Your task:
- Translate the following {source_lang.upper()} text to {target_lang.upper()} (Urdu)
- This is educational content for students learning robotics
- Maintain clarity, accuracy, and teaching quality

CRITICAL RULES:
1. **Preserve ALL placeholders** (e.g., __CODE_BLOCK_0__, __MATH_5__, __TERM_3__)
   - These contain code, math, and technical terms
   - Return them EXACTLY as they appear, in the SAME positions

2. **Educational tone**
   - Use formal, clear Urdu appropriate for textbooks
   - Explain concepts simply but accurately
   - Maintain the pedagogical structure

3. **Technical accuracy**
   - Preserve technical meaning precisely
   - Use established Urdu technical terms where they exist
   - For new terms, use English with Urdu explanation

4. **Formatting**
   - Keep markdown formatting (##, **, etc.)
   - Preserve line breaks and structure
   - Maintain list formatting (1., 2., -, *)

5. **Context-aware translation**"""

            if context.get("has_code"):
                prompt += "\n   - This content includes code examples - translate explanations, not code"

            if context.get("has_math"):
                prompt += "\n   - This content includes mathematical formulas - explain in Urdu, don't translate formulas"

            if context["content_type"] == "theory":
                prompt += "\n   - This is theoretical content - prioritize clarity of concepts"
            elif context["content_type"] == "practical":
                prompt += "\n   - This is practical/code content - emphasize implementation steps"

            prompt += "\n\nTranslate naturally while following ALL rules above."

        else:
            # Generic prompt for other languages
            prompt = f"Translate from {source_lang} to {target_lang}. Preserve placeholders like __CODE_BLOCK_0__."

        return prompt

    def _reassemble(
        self,
        translated_text: str,
        preserved_elements: Dict[str, str]
    ) -> str:
        """
        Restore preserved elements (code, math, terms) into translated text.
        """
        result = translated_text

        for placeholder, original in preserved_elements.items():
            result = result.replace(placeholder, original)

        return result

    def _validate_translation(self, original: str, translated: str):
        """
        Basic validation checks.

        Checks:
        - All code blocks present
        - All math formulas present
        - Markdown structure preserved
        """
        # Count code blocks
        original_code_blocks = original.count('```')
        translated_code_blocks = translated.count('```')

        if original_code_blocks != translated_code_blocks:
            print(f"⚠️  Warning: Code block count mismatch ({original_code_blocks} vs {translated_code_blocks})")

        # Count math blocks
        original_math = original.count('$$')
        translated_math = translated.count('$$')

        if original_math != translated_math:
            print(f"⚠️  Warning: Math block count mismatch ({original_math} vs {translated_math})")

        # Check if placeholders were removed (should all be replaced)
        if "__CODE_BLOCK_" in translated or "__MATH_" in translated:
            print("⚠️  Warning: Unreplaced placeholders detected")

    def _estimate_difficulty(self, content: str) -> str:
        """Estimate content difficulty (beginner/intermediate/advanced)."""
        # Count technical terms
        technical_term_count = sum(
            1 for term in self.preserve_terms
            if term.lower() in content.lower()
        )

        # Count math formulas
        math_count = content.count('$$') + content.count('$')

        # Simple heuristic
        if technical_term_count > 10 or math_count > 5:
            return "advanced"
        elif technical_term_count > 5 or math_count > 2:
            return "intermediate"
        else:
            return "beginner"

    def _classify_content_type(self, content: str) -> str:
        """Classify content as theory, practical, or mixed."""
        has_code = "```" in content
        has_math = "$$" in content

        if has_code and not has_math:
            return "practical"
        elif has_math and not has_code:
            return "theory"
        elif has_code and has_math:
            return "mixed"
        else:
            return "general"
```

## Caching Strategy

### Database-Backed Cache

```python
# Database model for translation cache
class TranslationCache(Base):
    __tablename__ = "translation_cache"

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    content_hash = Column(String, unique=True, index=True)  # SHA-256 of source
    source_language = Column(String, default="en")
    target_language = Column(String)
    translated_content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    hit_count = Column(Integer, default=0)  # Track cache hits

# Caching logic
import hashlib

def content_hash(text: str) -> str:
    """Generate hash for cache key."""
    return hashlib.sha256(text.encode()).hexdigest()

async def get_or_translate(
    content: str,
    translator: EducationalTranslator,
    target_lang: str = "ur"
) -> tuple[str, bool]:
    """
    Get translation from cache or translate and cache.

    Returns: (translated_content, was_cached)
    """
    hash_key = content_hash(content)

    # Check cache
    cached = await db.query(TranslationCache).filter(
        TranslationCache.content_hash == hash_key,
        TranslationCache.target_language == target_lang
    ).first()

    if cached:
        # Update hit count
        cached.hit_count += 1
        await db.commit()
        return cached.translated_content, True

    # Translate
    translated = await translator.translate_content(content, target_lang=target_lang)

    # Cache
    cache_entry = TranslationCache(
        content_hash=hash_key,
        source_language="en",
        target_language=target_lang,
        translated_content=translated
    )
    db.add(cache_entry)
    await db.commit()

    return translated, False
```

### Cache Analytics

```python
async def analyze_cache_performance():
    """
    Analyze translation cache effectiveness.

    Metrics:
    - Cache hit rate
    - Most frequently translated content
    - Cost savings from caching
    """
    total_requests = await db.query(TranslationCache).count()
    total_hits = await db.query(func.sum(TranslationCache.hit_count)).scalar()

    # Hit rate
    cache_hit_rate = total_hits / (total_requests + total_hits)

    print(f"Cache Hit Rate: {cache_hit_rate:.1%}")
    print(f"Total unique translations: {total_requests}")
    print(f"Total cache hits: {total_hits}")

    # Cost savings (assuming $0.02 per translation)
    cost_saved = total_hits * 0.02
    print(f"💰 Cost saved by caching: ${cost_saved:.2f}")
```

## RTL (Right-to-Left) Handling

### Frontend Integration for Urdu

```typescript
// React component for RTL support

interface TranslatedContentProps {
  content: string;
  language: 'en' | 'ur';
}

function TranslatedContent({ content, language }: TranslatedContentProps) {
  return (
    <div
      dir={language === 'ur' ? 'rtl' : 'ltr'}
      lang={language}
      className={language === 'ur' ? 'urdu-font' : 'english-font'}
    >
      <Markdown>{content}</Markdown>
    </div>
  );
}

// CSS for Urdu typography
```css
.urdu-font {
  font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
  font-size: 1.2em;  /* Urdu often needs larger size */
  line-height: 2;     /* More spacing for readability */
}

/* Code blocks stay LTR even in RTL context */
.urdu-font pre, .urdu-font code {
  direction: ltr;
  text-align: left;
}
```
```

## Quality Assurance

### Translation Quality Checks

```python
def check_translation_quality(original: str, translated: str) -> Dict:
    """
    Automated quality checks for translation.

    Returns: {
        "passed": bool,
        "issues": List[str],
        "warnings": List[str]
    }
    """
    issues = []
    warnings = []

    # 1. Length check (translation shouldn't be wildly different)
    orig_words = len(original.split())
    trans_words = len(translated.split())
    length_ratio = trans_words / orig_words

    if length_ratio < 0.5 or length_ratio > 2.0:
        warnings.append(f"Length ratio unusual: {length_ratio:.2f}")

    # 2. Code block integrity
    if original.count('```') != translated.count('```'):
        issues.append("Code block count mismatch")

    # 3. Math formula integrity
    if original.count('$$') != translated.count('$$'):
        issues.append("Math formula count mismatch")

    # 4. Link integrity
    orig_links = len(re.findall(r'\[.*?\]\(.*?\)', original))
    trans_links = len(re.findall(r'\[.*?\]\(.*?\)', translated))

    if orig_links != trans_links:
        warnings.append("Link count changed")

    # 5. Header structure
    orig_headers = len(re.findall(r'^#+\s', original, re.MULTILINE))
    trans_headers = len(re.findall(r'^#+\s', translated, re.MULTILINE))

    if orig_headers != trans_headers:
        warnings.append("Header structure changed")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }
```

## Best Practices

1. **Use specialized prompts**: Educational content needs different prompts than general translation
2. **Preserve technical terms**: Don't translate standardized technical vocabulary
3. **Cache aggressively**: Same content requested by multiple users
4. **Validate translations**: Automated checks for structure preservation
5. **Handle RTL properly**: Special CSS/layout for Urdu
6. **Keep code LTR**: Even in RTL context, code stays left-to-right
7. **Test with native speakers**: Automated translation needs human review
8. **Version translations**: Update when source content changes
9. **Track cache metrics**: Measure cost savings and hit rates
10. **Provide fallback**: If translation fails, show original

## Cost Management

### Translation Cost Estimation

```python
def estimate_translation_cost(
    content: str,
    model: str = "gpt-4-turbo-preview"
) -> float:
    """
    Estimate cost for translating content.

    GPT-4 Turbo pricing (2025):
    - Input: $10 per 1M tokens
    - Output: $30 per 1M tokens
    """
    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-4")

    # Input tokens (source + system prompt ~500 tokens)
    input_tokens = len(encoding.encode(content)) + 500

    # Output tokens (assume 1.2x input for Urdu)
    output_tokens = int(input_tokens * 1.2)

    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * 10
    output_cost = (output_tokens / 1_000_000) * 30

    total_cost = input_cost + output_cost

    return total_cost

# Example: 4 chapters, 50k words total
# ~65k tokens = ~$0.65 per translation
# With 90% cache hit rate: ~$0.065 actual cost
```

## Testing

### Translation Test Suite

```python
# Test cases for translation quality

TEST_CASES = [
    {
        "name": "Code preservation",
        "input": """
        Calculate FK:
        ```python
        def fk(theta):
            return np.array([x, y])
        ```
        """,
        "assertions": [
            lambda t: "```python" in t,
            lambda t: "def fk(theta):" in t,
            lambda t: "np.array" in t
        ]
    },
    {
        "name": "Math preservation",
        "input": "The Jacobian is $J = \\frac{\\partial x}{\\partial \\theta}$",
        "assertions": [
            lambda t: "$J =" in t,
            lambda t: "\\frac{\\partial x}{\\partial \\theta}$" in t
        ]
    },
    {
        "name": "Technical term preservation",
        "input": "Forward kinematics uses Denavit-Hartenberg parameters",
        "assertions": [
            lambda t: "forward kinematics" in t.lower(),
            lambda t: "denavit-hartenberg" in t.lower()
        ]
    }
]

@pytest.mark.asyncio
async def test_translation_preserves_elements():
    translator = EducationalTranslator(openai_client)

    for test_case in TEST_CASES:
        translated = await translator.translate_content(
            test_case["input"],
            target_lang="ur"
        )

        for assertion in test_case["assertions"]:
            assert assertion(translated), \
                f"Failed assertion for test: {test_case['name']}"
```

This comprehensive skill covers all aspects of high-quality educational content translation!
