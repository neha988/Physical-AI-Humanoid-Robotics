---
name: content-adaptation
description: Patterns for personalizing educational content based on user background, learning style, and progress. Use when implementing personalization algorithms, creating content variants, adapting difficulty levels, building recommendation systems, or tracking user engagement.
---

# Content Adaptation Skill

## Overview
Patterns and algorithms for personalizing educational content based on user background, learning style, and progress. Ensures every learner receives content optimized for their experience level and preferences.

## Personalization Dimensions

### User Profile Attributes

```python
from pydantic import BaseModel
from typing import List, Literal

class UserProfile(BaseModel):
    """Complete user profile for personalization."""

    # Experience levels
    software_experience: Literal["beginner", "intermediate", "advanced"]
    hardware_experience: Literal["beginner", "intermediate", "advanced"]
    math_background: Literal["basic", "intermediate", "advanced"]

    # Learning preferences
    preferred_style: Literal["visual", "code", "theory"]
    learning_goals: List[str]  # ["build-robot", "research", "industry", "hobby"]

    # Progress tracking
    current_chapter: str
    completed_chapters: List[str]
    quiz_scores: dict  # {chapter_id: score}
    time_spent: dict  # {chapter_id: minutes}

    # Dynamic attributes (updated by system)
    inferred_level: str  # Overall skill level
    engagement_pattern: str  # How user interacts with content
    struggling_topics: List[str]  # Topics user finds difficult
```

## Adaptation Strategies

### 1. Content Variant Selection

**Multi-Variant Content Authoring**:

```markdown
## Forward Kinematics

<!-- [default] Basic explanation for all users -->
Forward kinematics calculates the end-effector position from joint angles.

<!-- [sw-focused] Software engineer perspective -->
**[Software Engineer's View]**:
Think of FK as a pure function: `position = fk(joint_angles)`. It's deterministic
and can be cached. Modern implementations use autodiff frameworks like JAX:

```python
import jax.numpy as jnp

def fk(theta):
    """FK with automatic differentiation."""
    x = L1 * jnp.cos(theta[0]) + L2 * jnp.cos(theta[0] + theta[1])
    return jnp.array([x, y])

# Jacobian automatically computed!
J = jax.jacobian(fk)
```

<!-- [hw-focused] Hardware engineer perspective -->
**[Hardware Engineer's View]**:
In physical robots, FK runs at 100-1000 Hz on the controller. Optimization matters:
- Use lookup tables for common poses
- Precompute trigonometric values
- Consider fixed-point arithmetic for embedded systems

<!-- [math-focused] Mathematics perspective -->
**[Mathematical Foundation]**:
FK is a Lie group homomorphism $f: \\mathbb{R}^n \\to SE(3)$:

$$f(\\theta) = \\exp(\\xi_1 \\theta_1) \\cdots \\exp(\\xi_n \\theta_n) M$$

where $\\xi_i \\in se(3)$ are twist coordinates.
```

### Content Selection Algorithm

```python
from typing import Dict, List
from enum import Enum

class ContentFocus(str, Enum):
    SOFTWARE = "sw-focused"
    HARDWARE = "hw-focused"
    MATH = "math-focused"
    DEFAULT = "default"

class ContentSelector:
    """Select appropriate content variant based on user profile."""

    def __init__(self):
        # Weights for different profile attributes
        self.weights = {
            "software_experience": 0.35,
            "hardware_experience": 0.35,
            "math_background": 0.20,
            "preferred_style": 0.10
        }

    def select_primary_focus(self, profile: UserProfile) -> ContentFocus:
        """
        Determine primary content focus for user.

        Logic:
        1. If strong in one area, focus there
        2. If balanced, use preferred_style
        3. Default to balanced/software (most common background)
        """
        experience_scores = {
            "software": self._experience_to_score(profile.software_experience),
            "hardware": self._experience_to_score(profile.hardware_experience),
            "math": self._experience_to_score(profile.math_background)
        }

        # Find strongest area
        max_area = max(experience_scores, key=experience_scores.get)
        max_score = experience_scores[max_area]

        # If one area significantly stronger (>1 level difference)
        if max_score >= 2.5:
            if max_area == "software":
                return ContentFocus.SOFTWARE
            elif max_area == "hardware":
                return ContentFocus.HARDWARE
            elif max_area == "math":
                return ContentFocus.MATH

        # Otherwise, use preferred style
        if profile.preferred_style == "code":
            return ContentFocus.SOFTWARE
        elif profile.preferred_style == "theory":
            return ContentFocus.MATH

        # Default
        return ContentFocus.DEFAULT

    def _experience_to_score(self, level: str) -> float:
        """Convert experience level to numeric score."""
        mapping = {
            "beginner": 1.0,
            "basic": 1.0,
            "intermediate": 2.0,
            "advanced": 3.0
        }
        return mapping.get(level, 1.0)

    def select_content_sections(
        self,
        all_sections: Dict[ContentFocus, str],
        profile: UserProfile
    ) -> str:
        """
        Assemble personalized content from variants.

        Returns: Combined content with primary variant + relevant supplements
        """
        primary_focus = self.select_primary_focus(profile)

        # Start with primary content
        content_parts = []

        # Always include default/introduction
        if ContentFocus.DEFAULT in all_sections:
            content_parts.append(all_sections[ContentFocus.DEFAULT])

        # Add primary focused content
        if primary_focus in all_sections:
            content_parts.append(all_sections[primary_focus])

        # Add supplementary content if beginner in any area
        if self._is_beginner_in_any_area(profile):
            # Add scaffolding
            scaffolding = self._generate_scaffolding(profile)
            content_parts.insert(1, scaffolding)  # After intro, before main

        # Add code examples if user prefers code
        if profile.preferred_style == "code" and primary_focus != ContentFocus.SOFTWARE:
            if ContentFocus.SOFTWARE in all_sections:
                content_parts.append("### Additional Code Examples")
                content_parts.append(all_sections[ContentFocus.SOFTWARE])

        return "\n\n".join(content_parts)

    def _is_beginner_in_any_area(self, profile: UserProfile) -> bool:
        """Check if user is beginner in any area."""
        return any([
            profile.software_experience == "beginner",
            profile.hardware_experience == "beginner",
            profile.math_background == "basic"
        ])

    def _generate_scaffolding(self, profile: UserProfile) -> str:
        """
        Generate scaffolding content for beginners.

        Scaffolding = additional context, prerequisites, simplified explanations
        """
        scaffolding_parts = ["### Background Concepts"]

        if profile.software_experience == "beginner":
            scaffolding_parts.append(
                "**New to programming?** This section uses Python. "
                "Variables store values, functions perform operations, "
                "and arrays hold multiple numbers."
            )

        if profile.hardware_experience == "beginner":
            scaffolding_parts.append(
                "**New to hardware?** Robots have joints (like your elbow) "
                "controlled by motors. Sensors measure joint angles."
            )

        if profile.math_background == "basic":
            scaffolding_parts.append(
                "**Math background:** This uses trigonometry (sin, cos) "
                "and matrices (grids of numbers). Don't worry - code examples "
                "show how to compute these."
            )

        return "\n\n".join(scaffolding_parts)
```

### 2. Difficulty Adjustment

**Progressive Disclosure**: Reveal complexity gradually

```python
class DifficultyAdapter:
    """Adjust content difficulty based on user performance."""

    def adjust_difficulty(
        self,
        base_content: str,
        profile: UserProfile,
        chapter_id: str
    ) -> str:
        """
        Adjust content difficulty based on user's performance and background.

        Strategies:
        - Hide advanced sections for beginners
        - Add challenges for advanced users
        - Simplify explanations for struggling users
        """
        # Calculate user's effective level for this chapter
        effective_level = self._calculate_effective_level(profile, chapter_id)

        # Adjust content
        if effective_level == "beginner":
            # Simplify
            content = self._add_beginner_elements(base_content)
            content = self._hide_advanced_sections(content)

        elif effective_level == "intermediate":
            # Standard content
            content = base_content

        elif effective_level == "advanced":
            # Add challenges
            content = self._add_advanced_challenges(base_content)
            content = self._add_research_pointers(content)

        return content

    def _calculate_effective_level(
        self,
        profile: UserProfile,
        chapter_id: str
    ) -> str:
        """
        Determine user's effective level for this chapter.

        Considers:
        - Stated experience levels
        - Performance on previous chapters
        - Time spent (struggling? -> lower effective level)
        """
        # Base level from profile
        avg_experience = (
            self._exp_to_num(profile.software_experience) +
            self._exp_to_num(profile.hardware_experience) +
            self._exp_to_num(profile.math_background)
        ) / 3

        # Adjust based on performance
        if chapter_id in profile.quiz_scores:
            score = profile.quiz_scores[chapter_id]

            if score < 0.6:  # Struggling
                avg_experience = max(1.0, avg_experience - 0.5)
            elif score > 0.9:  # Excelling
                avg_experience = min(3.0, avg_experience + 0.5)

        # Map to level
        if avg_experience < 1.5:
            return "beginner"
        elif avg_experience < 2.5:
            return "intermediate"
        else:
            return "advanced"

    def _exp_to_num(self, exp: str) -> float:
        mapping = {"beginner": 1.0, "basic": 1.0, "intermediate": 2.0, "advanced": 3.0}
        return mapping.get(exp, 1.0)

    def _add_beginner_elements(self, content: str) -> str:
        """Add beginner-friendly elements."""
        # Add visual aids
        # Add step-by-step breakdowns
        # Add analogies
        return content + "\n\n### Visual Guide\n[Diagram showing step-by-step process]"

    def _hide_advanced_sections(self, content: str) -> str:
        """Remove advanced sections (marked with [advanced] tag)."""
        # Remove sections marked <!-- [advanced] -->
        return re.sub(r'<!-- \[advanced\] -->[\s\S]*?(?=##|\Z)', '', content)

    def _add_advanced_challenges(self, content: str) -> str:
        """Add advanced exercises and open-ended problems."""
        challenges = """
        ### Advanced Challenges

        1. **Extend the implementation**: Add support for 6-DOF manipulators
        2. **Optimize performance**: Reduce FK computation time to < 1ms
        3. **Research question**: How would you handle kinematic redundancy?
        """
        return content + "\n\n" + challenges

    def _add_research_pointers(self, content: str) -> str:
        """Add pointers to research papers and advanced topics."""
        return content + "\n\n### Further Reading\n- Recent ICRA/RSS papers on [topic]"
```

### 3. Learning Style Adaptation

```python
class StyleAdapter:
    """Adapt content presentation based on learning style."""

    def adapt_to_style(
        self,
        content: str,
        preferred_style: str
    ) -> str:
        """
        Adjust content presentation for learning style.

        Styles:
        - visual: Emphasize diagrams, flowcharts, visualizations
        - code: Lead with code examples, practical implementations
        - theory: Emphasize concepts, proofs, mathematical rigor
        """
        if preferred_style == "code":
            return self._code_first_presentation(content)
        elif preferred_style == "visual":
            return self._visual_first_presentation(content)
        elif preferred_style == "theory":
            return self._theory_first_presentation(content)
        else:
            return content  # Default order

    def _code_first_presentation(self, content: str) -> str:
        """
        Reorder content to show code first, theory after.

        Structure:
        1. Quick code example (working implementation)
        2. "How it works" explanation
        3. Theory and math (optional deeper dive)
        """
        # Parse content sections
        sections = self._parse_sections(content)

        # Reorder: code → explanation → theory
        reordered = []

        # Find and add code sections first
        code_sections = [s for s in sections if "```" in s["content"]]
        reordered.extend(code_sections)

        # Then explanations
        explanation_sections = [s for s in sections if s not in code_sections]
        reordered.extend(explanation_sections)

        return self._reassemble_sections(reordered)

    def _visual_first_presentation(self, content: str) -> str:
        """
        Emphasize visual elements.

        Additions:
        - ASCII diagrams
        - Flowcharts
        - Step-by-step visual breakdowns
        - Highlighted key points
        """
        # Add visual elements
        visual_intro = """
        ### Visual Overview
        ```
        Input: Joint Angles        Output: Position
           [θ₁, θ₂, θ₃]    →  [FK]  →  [x, y, z]
              ↓                          ↓
         Robot joints              End-effector
        ```
        """

        return visual_intro + "\n\n" + content

    def _theory_first_presentation(self, content: str) -> str:
        """
        Emphasize theoretical foundations.

        Structure:
        1. Formal definitions
        2. Mathematical derivations
        3. Proofs and properties
        4. Then practical implementation
        """
        # Reorder to emphasize theory
        # Add mathematical rigor
        return content  # Implement reordering logic
```

## Dynamic Adaptation

### Adaptive Learning Paths

```python
class LearningPathAdapter:
    """Adjust learning path based on user performance."""

    def recommend_next_content(
        self,
        profile: UserProfile,
        current_chapter: str
    ) -> Dict:
        """
        Recommend what to study next based on performance.

        Returns:
        {
            "next_chapter": str,
            "optional_reviews": List[str],
            "challenge_problems": List[str],
            "reason": str
        }
        """
        # Analyze performance on current chapter
        if current_chapter in profile.quiz_scores:
            score = profile.quiz_scores[current_chapter]

            if score < 0.7:
                # Struggling - recommend review
                return {
                    "next_chapter": current_chapter,  # Review current
                    "optional_reviews": self._find_prerequisite_topics(current_chapter),
                    "challenge_problems": [],
                    "reason": f"Review recommended (score: {score:.0%}). Master this before moving on."
                }

            elif score > 0.9:
                # Excelling - can skip ahead or take advanced path
                return {
                    "next_chapter": self._get_next_chapter(current_chapter),
                    "optional_reviews": [],
                    "challenge_problems": self._get_advanced_problems(current_chapter),
                    "reason": f"Great work! (score: {score:.0%}) Ready for advanced challenges."
                }

        # Standard progression
        return {
            "next_chapter": self._get_next_chapter(current_chapter),
            "optional_reviews": [],
            "challenge_problems": [],
            "reason": "Continue to next chapter"
        }

    def _find_prerequisite_topics(self, chapter: str) -> List[str]:
        """Find topics to review based on current chapter."""
        prerequisites = {
            "chapter-2": ["chapter-1", "linear-algebra-basics"],
            "chapter-3": ["chapter-2", "python-numpy"],
            "chapter-4": ["chapter-3", "neural-networks-intro"]
        }
        return prerequisites.get(chapter, [])

    def _get_next_chapter(self, current: str) -> str:
        """Get next chapter in sequence."""
        chapter_num = int(current.split('-')[1])
        return f"chapter-{chapter_num + 1}"

    def _get_advanced_problems(self, chapter: str) -> List[str]:
        """Get challenge problems for advanced learners."""
        # Fetch from database or predefined set
        return [
            "implement-6dof-fk",
            "optimize-jacobian-computation",
            "handle-kinematic-singularities"
        ]
```

## RAG Response Personalization

### Personalize Chatbot Answers

```python
async def personalize_rag_response(
    base_answer: str,
    user_profile: UserProfile,
    query_context: Dict
) -> str:
    """
    Adapt RAG chatbot response to user's level and style.

    Adjustments:
    - Technical depth
    - Example type (code vs hardware vs theory)
    - Explanation style
    - References to past progress
    """
    # Determine user's effective level
    level = determine_effective_level(user_profile)

    # Add personalized prefix based on level
    if level == "beginner":
        prefix = "(Simplified explanation) "
        # Could also: simplify the base answer using LLM
    elif level == "advanced":
        prefix = "(Technical detail) "
        # Could also: add more depth using LLM
    else:
        prefix = ""

    # Add style-specific examples
    if user_profile.preferred_style == "code":
        # Add code example if not present
        if "```" not in base_answer:
            base_answer += "\n\n```python\n# Example implementation\n...\n```"

    # Reference user's progress
    if query_context.get("chapter") in user_profile.completed_chapters:
        suffix = f"\n\nNote: You've completed {query_context['chapter']}. Want to move to advanced topics?"
    else:
        suffix = ""

    return prefix + base_answer + suffix
```

## Evaluation Metrics

### Measure Personalization Effectiveness

```python
class PersonalizationMetrics:
    """Track and evaluate personalization quality."""

    async def measure_effectiveness(self, user_id: str) -> Dict:
        """
        Measure how well personalization is working for a user.

        Metrics:
        - Engagement: time spent, pages viewed
        - Performance: quiz scores, problem completion
        - Satisfaction: explicit feedback, content ratings
        - Progress velocity: chapters per week
        """
        user = await get_user(user_id)
        profile = user.profile

        # Engagement metrics
        avg_time_per_chapter = np.mean(list(profile.time_spent.values()))
        chapters_completed = len(profile.completed_chapters)

        # Performance metrics
        avg_quiz_score = np.mean(list(profile.quiz_scores.values()))

        # Progress velocity
        days_since_start = (datetime.now() - user.created_at).days
        chapters_per_week = chapters_completed / (days_since_start / 7)

        return {
            "engagement": {
                "avg_time_per_chapter_min": avg_time_per_chapter,
                "chapters_completed": chapters_completed
            },
            "performance": {
                "avg_quiz_score": avg_quiz_score
            },
            "progress": {
                "chapters_per_week": chapters_per_week
            }
        }

    async def compare_with_without_personalization(self):
        """
        A/B test: measure impact of personalization.

        Compare control (no personalization) vs treatment (personalized).
        """
        # Get users with personalization
        treatment_users = await get_users(personalization_enabled=True)

        # Get users without personalization
        control_users = await get_users(personalization_enabled=False)

        # Compare metrics
        treatment_scores = [u.avg_quiz_score for u in treatment_users]
        control_scores = [u.avg_quiz_score for u in control_users]

        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(treatment_scores, control_scores)

        return {
            "treatment_mean": np.mean(treatment_scores),
            "control_mean": np.mean(control_scores),
            "improvement": (np.mean(treatment_scores) - np.mean(control_scores)) / np.mean(control_scores),
            "statistically_significant": p_value < 0.05
        }
```

## Best Practices

1. **Explicit > Implicit**: Ask users their preferences (background questionnaire)
2. **Start Simple**: Basic personalization first, add complexity gradually
3. **Measure Impact**: A/B test to verify personalization helps
4. **Allow Overrides**: Let users choose variant even if not "recommended"
5. **Progressive Enhancement**: Default content works without personalization
6. **Avoid Filter Bubbles**: Expose users to diverse content types
7. **Update Profiles**: User's level changes over time
8. **Respect Privacy**: Clear consent, data usage transparency
9. **Performance**: Personalization shouldn't slow down page loads
10. **Feedback Loops**: Use user behavior to improve personalization

## Anti-Patterns

❌ **Over-personalization**: Hiding too much content
❌ **No fallback**: Broken experience if personalization fails
❌ **Ignoring feedback**: User explicitly wants different content
❌ **Static profiles**: Not updating based on performance
❌ **Binary choices**: "Beginner or Advanced" → Use gradients
❌ **No transparency**: User doesn't know why they see this content
❌ **Performance degradation**: Slow page loads due to personalization logic
❌ **Forgetting prerequisites**: Jumping skill levels too fast

This skill provides a complete framework for content personalization!
