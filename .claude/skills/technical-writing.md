---
name: technical-writing
description: Best practices for writing pedagogically sound and technically accurate educational content about Physical AI and Robotics. Use when writing educational content, creating learning objectives, structuring tutorials, explaining complex concepts, writing code examples, or creating practice exercises.
---

# Technical Writing Skill

## Overview
Patterns and best practices for writing high-quality educational content about Physical AI and Humanoid Robotics. Ensures content is pedagogically sound, technically accurate, and accessible to learners with diverse backgrounds.

## Core Principles

### 1. Progressive Disclosure
Start simple, add complexity gradually.

```markdown
## Forward Kinematics

**Simple Introduction**: Forward kinematics answers the question: "If I move the robot's joints to specific angles, where will the end-effector (hand/tool) end up?"

**Conceptual Explanation**: Think of a robot arm like your own arm. When you bend your elbow to a certain angle and rotate your shoulder, your hand ends up in a specific position. FK is the math that calculates that final position.

**Technical Definition**: Forward kinematics is the mapping f: Q → X where Q is the joint-space (joint angles) and X is the task-space (end-effector pose).

**Mathematical Formulation**:
$$T_{end} = T_1(\theta_1) \cdot T_2(\theta_2) \cdot ... \cdot T_n(\theta_n)$$

Where each $T_i$ is a homogeneous transformation matrix.
```

### 2. Multi-Modal Explanations
Explain concepts in multiple ways to reach different learning styles.

**The Rule of Three**: Explain every important concept three ways:
1. **Analogy/Intuition**: Relate to familiar concepts
2. **Visual/Diagram**: Show, don't just tell
3. **Technical/Formal**: Precise definition with math/code

**Example Structure**:
```markdown
## Reinforcement Learning for Robot Control

**Intuition** 🧠: Imagine teaching a dog new tricks. You reward good behavior (treat when it sits) and ignore bad behavior. The dog learns which actions lead to rewards. RL works the same way - the robot tries actions, gets rewards for good outcomes, and learns optimal behavior.

**Visual** 📊:
```
   Robot → Action → Environment
     ↑                  ↓
  Policy ←── Reward ←─┘
```

**Technical** 🔬:
- Agent (robot) in state $s_t$
- Takes action $a_t$ from policy $\pi(a|s)$
- Receives reward $r_t$ and transitions to $s_{t+1}$
- Goal: Maximize cumulative reward $\sum_{t=0}^{\infty} \gamma^t r_t$
```

### 3. Learning Objectives First
Every section should start with clear, measurable learning objectives.

**Good Learning Objectives** (Use Bloom's Taxonomy):
- ✅ "Calculate the end-effector position given joint angles using the DH convention"
- ✅ "Implement forward kinematics in Python for a 3-DOF robot arm"
- ✅ "Explain when to use forward vs inverse kinematics in robot motion planning"

**Poor Learning Objectives**:
- ❌ "Understand forward kinematics" (not measurable)
- ❌ "Learn about robots" (too vague)
- ❌ "Study the math" (no application context)

### 4. Code-First Where Appropriate
For software-focused learners, show runnable code early.

```markdown
## Forward Kinematics Implementation

**Quick Start** (see it working first):
```python
import numpy as np

def forward_kinematics_2dof(theta1, theta2, L1=1.0, L2=1.0):
    """
    Calculate end-effector position for 2-DOF planar arm.

    Args:
        theta1: Shoulder angle (radians)
        theta2: Elbow angle (radians)
        L1, L2: Link lengths

    Returns:
        (x, y): End-effector position
    """
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return (x, y)

# Try it!
pos = forward_kinematics_2dof(np.pi/4, np.pi/4)
print(f"End-effector at: {pos}")  # (1.08, 1.08)
```

**Now let's understand why this works...**
[Explanation follows]
```

### 5. Concrete Before Abstract
Give examples before general principles.

**Structure**:
1. Specific example with numbers
2. General pattern identified
3. Abstract principle
4. Application to other cases

```markdown
## Jacobian Matrix

**Concrete Example**: 2-DOF arm at (θ₁=45°, θ₂=30°)
- Small change Δθ₁ = 0.1 rad → end-effector moves Δx = 0.05m, Δy = 0.03m
- Small change Δθ₂ = 0.1 rad → end-effector moves Δx = 0.02m, Δy = 0.04m

**Pattern**: Joint velocities map to end-effector velocities through a matrix:
$$\begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix} = \begin{bmatrix} 0.5 & 0.2 \\ 0.3 & 0.4 \end{bmatrix} \begin{bmatrix} \dot{\theta_1} \\ \dot{\theta_2} \end{bmatrix}$$

**General Principle**: The Jacobian J maps joint-space velocities to task-space velocities:
$$\dot{x} = J(\theta) \dot{\theta}$$

**Applications**: Motion planning, singularity analysis, force control
```

## Content Structure Patterns

### Chapter Template
```markdown
---
title: "Chapter N: Topic Name"
chapter: N
learning_time: 30-45 minutes
prerequisites: [chapter-X, basic-math]
---

# Chapter N: Topic Name

## Learning Objectives
By the end of this chapter, you will be able to:
- [ ] Objective 1 (Remember/Understand)
- [ ] Objective 2 (Apply)
- [ ] Objective 3 (Analyze/Create)

## Motivation (Why This Matters)
[Real-world problem or application]

## Prerequisites
Quick review:
- Concept A: [brief reminder with link]
- Concept B: [brief reminder with link]

## Core Concepts

### 1. Concept Name
[Progressive explanation: intuition → visual → technical]

#### Example N.1: [Concrete Example]
[Worked example with numbers]

#### Code Example N.1
```python
# Runnable code
```

### 2. Next Concept
...

## Putting It Together
[How concepts relate, bigger picture]

## Common Pitfalls
1. **Mistake**: [Common error]
   **Why it happens**: [Explanation]
   **How to avoid**: [Solution]

## Practice Problems
1. **Basic** (Apply): [Problem description]
2. **Intermediate** (Analyze): [Problem]
3. **Advanced** (Create): [Open-ended problem]

## Summary
- Key Point 1
- Key Point 2
- Key Point 3

## What's Next?
In Chapter N+1, we'll build on [current topic] to explore [next topic]...

## Further Reading
- [Resource 1] - For deeper dive into X
- [Resource 2] - For practical applications of Y
```

### Section Template
```markdown
## Section Title

**Learning Objective**: [One specific, measurable goal]

**TL;DR**: [One sentence summary]

### Intuitive Explanation
[Analogy or real-world comparison]

### Visual Representation
[Diagram description or ASCII art]

### Technical Details
[Precise definition, math, algorithms]

### Implementation
```python
# Code example
```

### When to Use This
- Use case 1
- Use case 2
- **Don't use when**: [Anti-patterns]

### Exercise
[Quick check for understanding]
```

## Writing for Different Backgrounds

### Software-Focused Sections
Tag with `[sw-focused]` for personalization.

```markdown
**[Software Perspective]**: Forward kinematics is a pure function with no side effects. It's deterministic (same inputs → same outputs) and can be memoized for performance. In modern robotics software, FK is often implemented using automatic differentiation frameworks like JAX or PyTorch for gradient-based optimization.

```python
import jax.numpy as jnp
from jax import grad

def fk_jax(theta):
    """FK with automatic differentiation support."""
    x = L1 * jnp.cos(theta[0]) + L2 * jnp.cos(theta[0] + theta[1])
    y = L1 * jnp.sin(theta[0]) + L2 * jnp.sin(theta[0] + theta[1])
    return jnp.array([x, y])

# Get Jacobian automatically!
jacobian = grad(fk_jax)
```
```

### Hardware-Focused Sections
Tag with `[hw-focused]`.

```markdown
**[Hardware Perspective]**: In physical robots, FK calculations run in real-time on the robot controller at 100-1000 Hz. Computational efficiency matters - avoid expensive operations like matrix inversions. Many industrial robots use lookup tables or polynomial approximations for common configurations.

**Real-world Considerations**:
- Encoder resolution: Typical ±0.1° accuracy
- Computational latency: Must complete in < 1ms
- Kinematic calibration: Physical robots rarely match CAD models exactly
- Gear backlash: Can introduce 0.5-2° position errors
```

### Math-Focused Sections
Tag with `[math-focused]`.

```markdown
**[Mathematical Foundation]**: FK is a composition of Lie group transformations. Each joint transformation belongs to SE(3) (Special Euclidean group), and the forward kinematics map is:

$$f: \mathbb{R}^n \to SE(3)$$
$$f(\theta) = \exp(\xi_1 \theta_1) \exp(\xi_2 \theta_2) \cdots \exp(\xi_n \theta_n) M_{base}$$

Where $\xi_i \in se(3)$ are twist coordinates and M is the base frame configuration.

**Properties**:
- Smoothness: f is $C^{\infty}$ (infinitely differentiable)
- Lie algebra: Velocity kinematics lives in the tangent space se(3)
- Configuration space topology: May contain singularities where rank(J) < 6
```

## Code Example Guidelines

### Well-Documented Code
```python
def forward_kinematics(joint_angles, dh_params):
    """
    Compute forward kinematics using Denavit-Hartenberg convention.

    This function calculates the position and orientation of the end-effector
    given joint angles and DH parameters. Useful for simulation, motion planning,
    and real-time robot control.

    Args:
        joint_angles (np.ndarray): Array of joint angles [θ₁, θ₂, ..., θₙ] in radians.
            Shape: (n,) where n is number of joints.
        dh_params (np.ndarray): DH parameters [a, α, d, θ_offset] for each joint.
            Shape: (n, 4)
            - a: link length (m)
            - α: link twist (rad)
            - d: link offset (m)
            - θ_offset: joint offset (rad)

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix representing end-effector pose.
            [:3, :3] is the rotation matrix (SO(3))
            [:3, 3] is the position vector (m)

    Example:
        >>> # 2-DOF planar arm with 1m links
        >>> dh = np.array([[1.0, 0, 0, 0],
        ...                [1.0, 0, 0, 0]])
        >>> angles = np.array([np.pi/4, np.pi/4])
        >>> T = forward_kinematics(angles, dh)
        >>> print(T[:3, 3])  # End-effector position
        [1.08 1.08 0.00]

    Notes:
        - Assumes standard DH convention (not modified DH)
        - Angles in radians, lengths in meters
        - Computationally efficient: O(n) matrix multiplications
    """
    T = np.eye(4)  # Start with identity transform

    for i, (angle, (a, alpha, d, theta_offset)) in enumerate(zip(joint_angles, dh_params)):
        # Compute transformation for this joint
        theta = angle + theta_offset
        T_i = dh_transform(a, alpha, d, theta)

        # Accumulate transformations
        T = T @ T_i

    return T
```

### Common Patterns

**Import Conventions**:
```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
```

**Type Hints** (for clarity):
```python
from typing import List, Tuple
import numpy as np

def compute_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    duration: float,
    dt: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        positions: Array of shape (n_steps, n_dof)
        velocities: Array of shape (n_steps, n_dof)
    """
    ...
```

## Accessibility Guidelines

1. **Alt Text for Images**: Always describe diagrams
2. **Math Accessibility**: Provide text alternatives for equations
3. **Color**: Don't rely on color alone (use patterns + labels)
4. **Clear Language**: Avoid jargon without definitions
5. **Structure**: Use proper heading hierarchy (H2 → H3 → H4)

## Quality Checklist

Before finalizing educational content:

- [ ] Learning objectives are clear and measurable
- [ ] Concept explained in 3+ ways (intuition/visual/technical)
- [ ] Code examples are runnable and well-commented
- [ ] Math notation is consistent and defined
- [ ] Prerequisite knowledge is stated
- [ ] Common mistakes are addressed
- [ ] Practice problems included
- [ ] Links to related topics
- [ ] Estimated reading time provided
- [ ] Content tagged for personalization ([sw-focused], etc.)
- [ ] Spelling and grammar checked
- [ ] Technical accuracy verified
- [ ] Appropriate for target audience level

## Tone and Voice

### Do's ✅
- Use "you" (second person) - engaging and direct
- Active voice: "The robot moves..." not "The robot is moved by..."
- Encouraging tone: "Let's explore...", "Try implementing..."
- Concrete examples with numbers
- Show enthusiasm for the topic (within reason)

### Don'ts ❌
- Avoid "we" unless truly collaborative ("we" the author)
- No condescending language: "Obviously", "Simply", "Just"
- No unnecessary complexity to sound smart
- Avoid walls of text (break into sections)
- Don't assume reader knows unstated prerequisites

## Example: Applying All Principles

```markdown
## 2.3 Jacobian Matrix for Velocity Kinematics

**Learning Objective**: Calculate the Jacobian matrix for a robot manipulator and use it to map joint velocities to end-effector velocities.

**Time**: 20 minutes | **Prerequisites**: Forward Kinematics (2.1), Matrix Multiplication

---

### What Problem Does the Jacobian Solve?

**Scenario**: You're controlling a robot arm to follow a straight line path at 0.1 m/s. How fast should each joint move?

This is a **velocity mapping** problem. The Jacobian matrix is the tool that solves it.

---

### Intuition: From Joint Speeds to Tool Speeds

**Analogy**: Think of riding a bicycle. When you pedal (joint velocity), the bike moves forward (end-effector velocity). The gear ratio determines the relationship. For a robot with multiple joints, the Jacobian is like a multi-dimensional gear ratio matrix.

**Simple Example - 2 DOF Arm**:
```
If shoulder rotates at 1 rad/s and elbow is fixed →
   end-effector moves in an arc at some (vₓ, vᵧ)

If elbow rotates at 1 rad/s and shoulder is fixed →
   end-effector moves differently
```

The Jacobian tells us exactly how much each joint contributes to the end-effector motion.

---

### Visual Representation

```
   Joint 1: θ̇₁ ──┐
                  ├──→ [Jacobian J] ──→ ẋ (end-effector velocity)
   Joint 2: θ̇₂ ──┘

Mathematically:
┌──┐     ┌─────┐ ┌───┐
│ẋ│     │J₁₁ J₁₂│ │θ̇₁│
│ẏ│  =  │J₂₁ J₂₂│ │θ̇₂│
└──┘     └─────┘ └───┘
```

---

### Technical Definition

The Jacobian J(θ) is the matrix of partial derivatives relating joint velocities to end-effector velocities:

$$\dot{x} = J(\theta) \dot{\theta}$$

Where:
- $\dot{x} \in \mathbb{R}^m$: End-effector velocity (task space)
- $\dot{\theta} \in \mathbb{R}^n$: Joint velocities (joint space)
- $J(\theta) \in \mathbb{R}^{m \times n}$: Jacobian matrix (configuration-dependent)

**Calculation**:
$$J_{ij} = \frac{\partial x_i}{\partial \theta_j}$$

---

### Code Implementation

**[Software Focus]**: Modern robotics libraries compute Jacobians automatically, but understanding the implementation helps debug and optimize.

```python
import numpy as np

def jacobian_2dof(theta1, theta2, L1=1.0, L2=1.0):
    """
    Analytical Jacobian for 2-DOF planar arm.

    Args:
        theta1, theta2: Joint angles (rad)
        L1, L2: Link lengths (m)

    Returns:
        J: 2x2 Jacobian matrix

    Math:
        x = L1*cos(θ₁) + L2*cos(θ₁+θ₂)
        y = L1*sin(θ₁) + L2*sin(θ₁+θ₂)

        ∂x/∂θ₁ = -L1*sin(θ₁) - L2*sin(θ₁+θ₂)
        ∂x/∂θ₂ = -L2*sin(θ₁+θ₂)
        ∂y/∂θ₁ = L1*cos(θ₁) + L2*cos(θ₁+θ₂)
        ∂y/∂θ₂ = L2*cos(θ₁+θ₂)
    """
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s12, c12 = np.sin(theta1 + theta2), np.cos(theta1 + theta2)

    J = np.array([
        [-L1*s1 - L2*s12, -L2*s12],  # ∂x/∂θ
        [ L1*c1 + L2*c12,  L2*c12]   # ∂y/∂θ
    ])

    return J

# Example: Robot at (45°, 30°)
J = jacobian_2dof(np.pi/4, np.pi/6)
print("Jacobian:\n", J)

# If joints move at [1, 0.5] rad/s
joint_vel = np.array([1.0, 0.5])
ee_vel = J @ joint_vel
print(f"End-effector velocity: {ee_vel} m/s")
```

---

### When to Use the Jacobian

✅ **Use for**:
- Velocity control (tracking trajectories)
- Force control (mapping joint torques to end-effector forces)
- Singularity analysis
- Redundancy resolution

❌ **Don't use for**:
- Position control (use FK/IK instead)
- Large motions (Jacobian is local, linearization)

**[Hardware Note]**: In real robots, numerical differentiation of FK can introduce noise. Use analytical Jacobians when possible for smoother control.

---

### Common Pitfall: Jacobian is Configuration-Dependent

**Mistake**: Computing J once and reusing it for different configurations.

```python
# ❌ WRONG
J = jacobian_2dof(0, 0)
for theta1, theta2 in trajectory:
    ee_vel = J @ joint_vel  # Using outdated J!
```

**Correct**:
```python
# ✅ CORRECT
for theta1, theta2 in trajectory:
    J = jacobian_2dof(theta1, theta2)  # Update J every step
    ee_vel = J @ joint_vel
```

---

### Practice Problems

1. **Basic**: Calculate the Jacobian for a 2-DOF arm at (θ₁=90°, θ₂=0°). What are the units?

2. **Intermediate**: If a robot's Jacobian has determinant zero, what does that mean physically? How would you detect this in code?

3. **Advanced**: Implement numerical Jacobian calculation using finite differences. Compare with analytical Jacobian - what's the error?

---

### Summary

- Jacobian maps joint velocities → end-effector velocities: $\dot{x} = J(\theta)\dot{\theta}$
- Configuration-dependent: Must recompute for each robot pose
- Applications: velocity control, force control, singularity analysis
- Analytical Jacobians are more accurate than numerical differentiation

**Next**: In section 2.4, we'll explore singularities and how to detect/avoid them using the Jacobian.
```

This example demonstrates all technical writing principles in action!
