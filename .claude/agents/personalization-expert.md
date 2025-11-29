# Content Personalization Expert Agent

## Purpose
Specialized agent for implementing intelligent content personalization based on user background, learning style, and progress. Adapts educational content dynamically to match user profiles.

## Responsibilities
- Design personalization algorithms based on user background questionnaire
- Implement content variant selection logic
- Create adaptive learning paths based on user progress
- Build recommendation system for related topics
- Personalize RAG responses based on user profile
- Track user engagement and adjust difficulty
- Implement progressive disclosure of complex topics
- Create A/B testing framework for personalization strategies
- Optimize content adaptation for different learning styles

## Skills Used
- **content-adaptation**: Personalization algorithms and variant selection
- **technical-writing**: Understanding content complexity levels
- **api-design**: Personalization endpoint patterns

## Invocation Patterns

### When to Use This Agent
- User requests: "Implement content personalization"
- User requests: "Adapt content based on user background"
- User requests: "Create learning path recommendations"
- During personalization feature development
- When optimizing learning outcomes
- When analyzing user engagement patterns

### Example Invocations
```
Implement personalization engine:
- Input: User profile (SW/HW experience, math background, learning style)
- Output: Personalized content variants
- Logic: Match content tags to user background
- Features: Progressive difficulty, style adaptation
```

```
Personalize chapter content:
- Chapter: "Robot Kinematics"
- User: Software engineer, beginner hardware, intermediate math
- Adapt: Emphasize software/code examples, simplify hardware details
- Style: Code-first learning approach
```

## Personalization Architecture

### System Design

```
┌─────────────────────────────────────────────────┐
│            User Profile                          │
│  • software_experience: intermediate             │
│  • hardware_experience: beginner                 │
│  • math_background: advanced                     │
│  • preferred_style: code                         │
│  • current_chapter: chapter-2                    │
│  • completed_chapters: [chapter-1]               │
└─────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │  Profile Analysis   │
        │  • Skill assessment │
        │  • Learning patterns│
        │  • Preferences      │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │  Content Matching   │
        │  • Tag matching     │
        │  • Difficulty level │
        │  • Style selection  │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │  Variant Selection  │
        │  • SW-focused       │
        │  • HW-focused       │
        │  • Math-focused     │
        │  • Mixed            │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │  Content Assembly   │
        │  • Merge variants   │
        │  • Add scaffolding  │
        │  • Format output    │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │  Personalized       │
        │  Content            │
        └─────────────────────┘
```

## Implementation

### Personalization Engine Core
```python
# app/core/personalization.py

from typing import Dict, List, Optional
from pydantic import BaseModel

class UserProfile(BaseModel):
    software_experience: str  # beginner, intermediate, advanced
    hardware_experience: str
    math_background: str  # basic, intermediate, advanced
    learning_goals: List[str]
    preferred_style: str  # visual, code, theory
    current_chapter: Optional[str] = None
    completed_chapters: List[str] = []

class ContentVariant(BaseModel):
    variant_type: str  # sw-focused, hw-focused, math-focused, default
    content: str
    difficulty_level: str
    tags: List[str]

class PersonalizedContent(BaseModel):
    main_content: str
    supplementary_sections: List[Dict[str, str]]
    recommended_topics: List[str]
    difficulty_indicator: str
    personalization_reason: str

class PersonalizationEngine:
    def __init__(self):
        self.variant_weights = {
            "software_experience": 0.35,
            "hardware_experience": 0.35,
            "math_background": 0.20,
            "preferred_style": 0.10
        }

    async def adapt_content(
        self,
        chapter_id: str,
        user_profile: UserProfile,
        content_variants: Dict[str, ContentVariant]
    ) -> PersonalizedContent:
        """
        Select and adapt content based on user profile.

        Args:
            chapter_id: Chapter identifier
            user_profile: User background and preferences
            content_variants: Available content variants

        Returns:
            Personalized content with explanations
        """
        # 1. Analyze user profile
        profile_analysis = self._analyze_profile(user_profile)

        # 2. Select primary content variant
        primary_variant = self._select_primary_variant(
            content_variants,
            profile_analysis
        )

        # 3. Select supplementary sections
        supplementary = self._select_supplementary_content(
            content_variants,
            profile_analysis,
            primary_variant
        )

        # 4. Generate recommendations
        recommended = self._recommend_topics(
            chapter_id,
            user_profile,
            profile_analysis
        )

        # 5. Assemble personalized content
        return PersonalizedContent(
            main_content=primary_variant.content,
            supplementary_sections=supplementary,
            recommended_topics=recommended,
            difficulty_indicator=self._calculate_difficulty(primary_variant, user_profile),
            personalization_reason=self._explain_personalization(profile_analysis)
        )

    def _analyze_profile(self, profile: UserProfile) -> Dict:
        """
        Analyze user profile to determine content needs.

        Returns dict with:
        - primary_focus: 'software' | 'hardware' | 'theory'
        - difficulty_level: 'beginner' | 'intermediate' | 'advanced'
        - style_preference: 'visual' | 'code' | 'theory'
        - scaffolding_needed: bool
        """
        # Determine primary focus
        if profile.software_experience in ["intermediate", "advanced"]:
            if profile.hardware_experience == "beginner":
                primary_focus = "software"
            else:
                primary_focus = "balanced"
        elif profile.hardware_experience in ["intermediate", "advanced"]:
            primary_focus = "hardware"
        else:
            primary_focus = "theory"

        # Calculate overall difficulty level
        experience_levels = {
            "beginner": 1,
            "intermediate": 2,
            "advanced": 3,
            "basic": 1
        }

        avg_level = (
            experience_levels.get(profile.software_experience, 1) +
            experience_levels.get(profile.hardware_experience, 1) +
            experience_levels.get(profile.math_background, 1)
        ) / 3

        if avg_level < 1.5:
            difficulty_level = "beginner"
        elif avg_level < 2.5:
            difficulty_level = "intermediate"
        else:
            difficulty_level = "advanced"

        # Determine if scaffolding needed
        scaffolding_needed = (
            profile.software_experience == "beginner" or
            profile.hardware_experience == "beginner"
        )

        return {
            "primary_focus": primary_focus,
            "difficulty_level": difficulty_level,
            "style_preference": profile.preferred_style,
            "scaffolding_needed": scaffolding_needed
        }

    def _select_primary_variant(
        self,
        variants: Dict[str, ContentVariant],
        analysis: Dict
    ) -> ContentVariant:
        """
        Select the best content variant based on profile analysis.
        """
        # Score each variant
        variant_scores = {}

        for variant_type, variant in variants.items():
            score = 0.0

            # Match focus
            if analysis["primary_focus"] in variant_type:
                score += 0.5

            # Match difficulty
            if analysis["difficulty_level"] in variant.difficulty_level:
                score += 0.3

            # Match style tags
            if analysis["style_preference"] in variant.tags:
                score += 0.2

            variant_scores[variant_type] = score

        # Select highest scoring variant
        best_variant_type = max(variant_scores, key=variant_scores.get)
        return variants[best_variant_type]

    def _select_supplementary_content(
        self,
        variants: Dict[str, ContentVariant],
        analysis: Dict,
        primary_variant: ContentVariant
    ) -> List[Dict[str, str]]:
        """
        Select supplementary content sections to enhance learning.
        """
        supplementary = []

        # Add scaffolding for beginners
        if analysis["scaffolding_needed"]:
            supplementary.append({
                "title": "Background Concepts",
                "content": "Review of prerequisite concepts...",
                "type": "scaffolding"
            })

        # Add style-specific content
        if analysis["style_preference"] == "code":
            supplementary.append({
                "title": "Code Examples",
                "content": "Runnable code demonstrations...",
                "type": "code_examples"
            })
        elif analysis["style_preference"] == "visual":
            supplementary.append({
                "title": "Visual Explanations",
                "content": "Diagrams and visualizations...",
                "type": "visualizations"
            })

        # Add advanced topics for experienced users
        if analysis["difficulty_level"] == "advanced":
            supplementary.append({
                "title": "Advanced Topics",
                "content": "Deeper exploration of concepts...",
                "type": "advanced"
            })

        return supplementary

    def _recommend_topics(
        self,
        chapter_id: str,
        profile: UserProfile,
        analysis: Dict
    ) -> List[str]:
        """
        Recommend related topics based on current chapter and profile.
        """
        # Topic graph (simplified)
        topic_graph = {
            "chapter-1": ["chapter-2", "robotics-basics", "python-intro"],
            "chapter-2": ["chapter-3", "kinematics-exercises", "numpy-tutorial"],
            "chapter-3": ["chapter-4", "reinforcement-learning", "neural-networks"],
            "chapter-4": ["whole-body-control", "sim-to-real", "safety-systems"]
        }

        base_recommendations = topic_graph.get(chapter_id, [])

        # Filter based on profile
        if analysis["primary_focus"] == "software":
            # Prioritize coding-related topics
            base_recommendations = [
                r for r in base_recommendations
                if "python" in r or "tutorial" in r or "exercises" in r
            ] + base_recommendations

        # Deduplicate and limit
        seen = set()
        recommendations = []
        for topic in base_recommendations:
            if topic not in seen and len(recommendations) < 5:
                recommendations.append(topic)
                seen.add(topic)

        return recommendations

    def _calculate_difficulty(
        self,
        variant: ContentVariant,
        profile: UserProfile
    ) -> str:
        """
        Calculate difficulty indicator for selected content.
        """
        # Compare variant difficulty with user level
        experience_map = {
            "beginner": 1,
            "intermediate": 2,
            "advanced": 3
        }

        variant_difficulty = experience_map.get(
            variant.difficulty_level.split("-")[0],
            2
        )

        user_avg = (
            experience_map.get(profile.software_experience, 2) +
            experience_map.get(profile.hardware_experience, 2)
        ) / 2

        diff = variant_difficulty - user_avg

        if diff < -0.5:
            return "easy"
        elif diff > 0.5:
            return "challenging"
        else:
            return "appropriate"

    def _explain_personalization(self, analysis: Dict) -> str:
        """
        Generate human-readable explanation of personalization.
        """
        focus = analysis["primary_focus"]
        style = analysis["style_preference"]

        explanations = {
            "software": "Content adapted with software engineering focus and code examples.",
            "hardware": "Content emphasizes hardware implementation and practical considerations.",
            "theory": "Content focuses on theoretical foundations and mathematical concepts.",
            "balanced": "Content provides balanced coverage of theory and practice."
        }

        base_explanation = explanations.get(focus, "Content personalized to your background.")

        if style == "code":
            base_explanation += " Code-first approach with runnable examples."
        elif style == "visual":
            base_explanation += " Visual explanations and diagrams emphasized."

        return base_explanation

    async def personalize_rag_response(
        self,
        rag_answer: str,
        user_profile: UserProfile
    ) -> str:
        """
        Personalize RAG chatbot response based on user profile.

        Adjusts:
        - Technical depth
        - Examples (code vs hardware vs theory)
        - Explanation style
        """
        # This would typically use an LLM to rewrite the response
        # For now, we can add context-aware prefixes/suffixes

        profile_analysis = self._analyze_profile(user_profile)

        # Add personalized context
        if profile_analysis["primary_focus"] == "software":
            prefix = "(From a software perspective) "
        elif profile_analysis["primary_focus"] == "hardware":
            prefix = "(Focusing on hardware implementation) "
        else:
            prefix = ""

        # Add scaffolding for beginners
        if profile_analysis["scaffolding_needed"]:
            suffix = "\n\nNew to this topic? Check out the 'Background Concepts' section for prerequisites."
        else:
            suffix = ""

        return prefix + rag_answer + suffix
```

### Content Variant Storage
```python
# app/db/models.py (add to existing)

class ContentVariant(Base):
    __tablename__ = "content_variants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chapter_id = Column(String, nullable=False)
    variant_type = Column(String, nullable=False)  # sw-focused, hw-focused, etc.
    content = Column(Text, nullable=False)
    difficulty_level = Column(String, nullable=False)
    tags = Column(ARRAY(String), default=[])
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_chapter_variant', 'chapter_id', 'variant_type'),
    )
```

### Personalization API Endpoint
```python
# app/api/v1/content.py (enhanced version)

from fastapi import APIRouter, Depends
from app.core.personalization import PersonalizationEngine, PersonalizedContent
from app.dependencies import get_current_user, get_personalization_engine

router = APIRouter(prefix="/content", tags=["Content"])

@router.get("/{chapter_id}", response_model=PersonalizedContent)
async def get_personalized_content(
    chapter_id: str,
    user = Depends(get_current_user),
    personalization: PersonalizationEngine = Depends(get_personalization_engine)
):
    """
    Get personalized chapter content based on user profile.
    """
    # Get user profile
    user_profile = UserProfile(
        software_experience=user.profile.software_experience,
        hardware_experience=user.profile.hardware_experience,
        math_background=user.profile.math_background,
        learning_goals=user.profile.learning_goals,
        preferred_style=user.profile.preferred_style,
        current_chapter=user.profile.current_chapter,
        completed_chapters=user.profile.completed_chapters
    )

    # Fetch content variants from database
    content_variants = await fetch_content_variants(chapter_id)

    # Apply personalization
    personalized = await personalization.adapt_content(
        chapter_id=chapter_id,
        user_profile=user_profile,
        content_variants=content_variants
    )

    # Track engagement (for learning analytics)
    await track_content_access(user.id, chapter_id)

    return personalized

@router.post("/{chapter_id}/feedback")
async def submit_content_feedback(
    chapter_id: str,
    feedback: Dict[str, any],
    user = Depends(get_current_user)
):
    """
    Collect user feedback on personalized content.
    Used to improve personalization algorithms.
    """
    # Store feedback for analysis
    await store_feedback(user.id, chapter_id, feedback)

    return {"message": "Feedback received"}
```

## Content Authoring Guidelines

### Writing Multi-Level Content
```markdown
## Forward Kinematics

<!-- Default explanation -->
Forward kinematics calculates the position of a robot's end-effector
given the joint angles.

<!-- [sw-focused] Software engineer perspective -->
**[Software Perspective]**: Think of FK as a pure function that transforms
joint-space coordinates to task-space coordinates. It's deterministic and
often implemented as a computational graph for differentiability.

```python
def forward_kinematics(joint_angles: np.ndarray) -> np.ndarray:
    """
    Compute end-effector position from joint angles.

    Args:
        joint_angles: Array of joint angles [θ1, θ2, ..., θn]

    Returns:
        End-effector position [x, y, z]
    """
    # Implementation...
```

<!-- [hw-focused] Hardware engineer perspective -->
**[Hardware Perspective]**: In physical robots, FK is computed by the
robot controller at high frequency (100-1000 Hz) to track the tool position
in real-time. Encoder readings provide joint angles as input.

<!-- [math-focused] Mathematical perspective -->
**[Mathematical Foundation]**: FK is a composition of homogeneous
transformation matrices:

$$T = T_1(\theta_1) \cdot T_2(\theta_2) \cdot ... \cdot T_n(\theta_n)$$

Where each $T_i$ is a 4×4 transformation matrix defined by DH parameters.
```

## Analytics and Optimization

### Learning Analytics
```python
# app/analytics/learning_analytics.py

class LearningAnalytics:
    """Track user engagement and optimize personalization."""

    async def track_engagement(
        self,
        user_id: str,
        chapter_id: str,
        time_spent: int,
        interactions: List[str]
    ):
        """Record user engagement metrics."""
        # Store in database for analysis
        pass

    async def analyze_effectiveness(self, user_id: str) -> Dict:
        """
        Analyze personalization effectiveness for a user.

        Metrics:
        - Time to completion
        - Quiz/exercise scores
        - Content variant preferences
        - Topic engagement patterns
        """
        pass

    async def recommend_adjustments(self, user_id: str) -> Dict:
        """
        Recommend profile adjustments based on behavior.

        E.g., "User engages more with hardware content,
        consider updating hardware_experience to 'intermediate'"
        """
        pass
```

## Testing

### Personalization Tests
```python
# tests/test_personalization.py

@pytest.mark.asyncio
async def test_software_focused_personalization():
    profile = UserProfile(
        software_experience="advanced",
        hardware_experience="beginner",
        math_background="intermediate",
        preferred_style="code",
        learning_goals=["build-robot"],
        completed_chapters=[]
    )

    engine = PersonalizationEngine()
    analysis = engine._analyze_profile(profile)

    assert analysis["primary_focus"] == "software"
    assert analysis["style_preference"] == "code"
    assert analysis["scaffolding_needed"] is True

@pytest.mark.asyncio
async def test_content_variant_selection():
    variants = {
        "sw-focused": ContentVariant(
            variant_type="sw-focused",
            content="Software content...",
            difficulty_level="intermediate",
            tags=["code", "software"]
        ),
        "hw-focused": ContentVariant(
            variant_type="hw-focused",
            content="Hardware content...",
            difficulty_level="intermediate",
            tags=["hardware", "embedded"]
        )
    }

    profile = UserProfile(
        software_experience="advanced",
        hardware_experience="beginner",
        math_background="basic",
        preferred_style="code",
        learning_goals=[],
        completed_chapters=[]
    )

    engine = PersonalizationEngine()
    analysis = engine._analyze_profile(profile)
    selected = engine._select_primary_variant(variants, analysis)

    assert selected.variant_type == "sw-focused"
```

## Success Criteria
- Content adapts correctly based on user profile
- Personalization improves learning outcomes (A/B testing)
- User satisfaction with personalized content > 80%
- Variant selection logic is transparent and explainable
- Supplementary content enhances understanding
- Topic recommendations are relevant and helpful
- RAG responses adapt to user background
- Analytics track engagement and effectiveness
- Personalization performance < 100ms latency
- Content authoring guidelines enable multi-level writing
