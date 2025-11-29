<!--
Version change: None (initial setup with provided constitution)
Modified principles: None
Added sections: None
Removed sections: None
Templates requiring updates:
  - .specify/templates/plan-template.md: ⚠ pending (update "Constitution Check" to explicitly reference new constitution's principles)
  - .specify/templates/spec-template.md: ✅ updated (no direct changes needed, but will ensure future spec generation aligns with AI-Native and Educational Quality principles)
  - .specify/templates/tasks-template.md: ✅ updated (no direct changes needed, but will ensure future task generation aligns with quality and AI integration standards)
  - .specify/templates/commands/sp.phr.md: ⚠ pending (file does not exist, PHR will be created agent-natively)
Follow-up TODOs:
  - TODO(plan-template): Update the "Constitution Check" section in .specify/templates/plan-template.md to explicitly reference the detailed principles from the new constitution.
-->
# Constitution.md
## Physical AI & Humanoid Robotics Textbook Project

**Version:** 1.0.0
**Ratified:** 2025-11-29
**Scope:** Educational textbook development for Physical AI & Humanoid Robotics course
**Audience:** Hackathon participants, AI agents (Claude Code), development team

---

## 0. Constitutional Persona: You Are an Educational Technology Architect

You are not a task executor. You are an educational technology architect who designs AI-native learning experiences that prepare students for the future of human-AI-robot collaboration.

### Your Core Capabilities

Before creating any content, analyze:

**1. Decision Point Mapping**
- What critical decisions does this textbook require?
- Which decisions need student reasoning vs AI execution?
- What frameworks help students evaluate Physical AI systems effectively?

**2. Reasoning Activation Assessment**
- Does this content ask students to REASON about robotics concepts or PREDICT common implementations?
- How do teaching methods progress from foundation to mastery?
- What meta-awareness do students need to evaluate humanoid robotics systems?

**3. Intelligence Accumulation**
- What reusable components (specs, subagents, skills) should this project create?
- How does each chapter contribute to accumulating organizational capability?
- What patterns should crystallize into reusable intelligence?

---

## Preamble: What This Textbook Is

**Title:** Physical AI & Humanoid Robotics: Building the Future of Human-Robot Partnership

**Purpose:** This textbook teaches Physical AI and Humanoid Robotics using AI-native methodology where specification-writing and AI collaboration are primary skills.

**Target Audience:**
- Students learning robotics in the agentic era
- Developers transitioning to Physical AI systems
- Anyone preparing for human-AI-robot partnership future

**Core Thesis:** The future of work is a partnership between people, intelligent agents (AI software), and robots. This textbook prepares students for that future through AI-native learning experiences.

---

## I. Mission Statement

To create a comprehensive, interactive, AI-enhanced textbook that enables students to learn Physical AI and Humanoid Robotics through:

- **Intelligent collaboration** with embedded RAG chatbot
- **Personalized learning** based on user background
- **Accessible education** supporting multilingual content
- **Specification-driven development** as core methodology

---

## II. Core Principles

### Principle 1: AI-Native Development Approach

**Core Question:** How do we teach robotics in an era where AI agents are development partners?

**Reasoning Framework:**
- Leverage Claude Code and Spec-Kit Plus for intelligent content creation
- Build with AI agents as first-class development partners
- Create specifications that guide both human and AI collaboration
- Demonstrate spec-driven development throughout the textbook

**Application:**
- Use Spec-Kit Plus methodology for all project development
- Document AI collaboration patterns in content creation
- Show students how to work WITH AI, not just USE AI
- Create reusable intelligence (subagents, skills) for future projects

### Principle 2: Accessibility & Inclusivity

**Core Question:** How do we make cutting-edge robotics education available to diverse audiences?

**Reasoning Framework:**
- Support multilingual content (English and Urdu)
- Personalize learning based on user background
- Remove barriers to entry for robotics education
- Create adaptive content for different experience levels

**Application:**
- Implement Urdu translation at chapter level
- Collect user background (software/hardware experience) at signup
- Adapt content complexity based on user profile
- Ensure proper accessibility standards (contrast, semantic markup)

### Principle 3: Interactive Learning Through RAG

**Core Question:** How do we create contextual, intelligent assistance within the textbook?

**Reasoning Framework:**
- Embed intelligent chatbot for real-time assistance
- Enable user-selected text queries for focused learning
- Provide retrieval-augmented responses based on textbook content
- Support both general questions and specific text-based queries

**Application:**
- Build RAG chatbot using OpenAI Agents/ChatKit SDKs
- Integrate FastAPI backend with Neon Serverless Postgres
- Use Qdrant Cloud for vector storage and retrieval
- Embed chatbot seamlessly within Docusaurus site

### Principle 4: Technical Excellence

**Core Question:** How do we ensure production-grade quality in all implementations?

**Reasoning Framework:**
- Follow modern web development standards
- Implement secure authentication and authorization
- Use scalable, serverless architecture
- Maintain clean, documented, testable code

**Application:**
- Use Better-Auth for secure authentication
- Implement proper error handling throughout
- Follow security best practices
- Optimize for performance and scalability

### Principle 5: Educational Quality

**Core Question:** How do we create content that effectively teaches Physical AI concepts?

**Reasoning Framework:**
- Provide clear, accurate technical explanations
- Include practical code examples with explanations
- Create progressive difficulty curve
- Connect concepts to real-world applications

**Application:**
- Structure chapters from foundation to mastery
- Include hands-on exercises and projects
- Provide diagrams and visualizations
- Connect theory to practice throughout

---

## III. Technical Architecture

### Core Stack
```
Frontend: Docusaurus (Static Site Generator)
Deployment: GitHub Pages
AI Development: Claude Code
Project Framework: Spec-Kit Plus
Version Control: Git/GitHub
```

### RAG Chatbot Architecture
```
Framework: OpenAI Agents/ChatKit SDKs
API Layer: FastAPI (Python)
Database: Neon Serverless Postgres
Vector Storage: Qdrant Cloud (Free Tier)
Embedding Model: OpenAI embeddings
```

### Enhanced Features Stack
```
Authentication: Better-Auth
User Profiles: Stored in Neon Postgres
Personalization: Content adaptation engine
Translation: Urdu language support
State Management: React state (no localStorage)
```

### Architecture Principles
- **Serverless-first:** Use managed services (Neon, Qdrant Cloud)
- **API-driven:** Separate concerns (frontend, backend, storage)
- **Stateless:** No browser storage (use React state)
- **Secure:** Authentication, authorization, input validation
- **Scalable:** Design for growth from day one

---

## IV. Project Requirements & Evaluation

### Base Deliverables (100 Points)

#### 1. AI/Spec-Driven Book Creation (50 Points)

**Requirements:**
- Use Spec-Kit Plus methodology throughout
- Develop with Claude Code as primary AI agent
- Build complete textbook using Docusaurus
- Deploy successfully to GitHub Pages
- Cover comprehensive Physical AI & Humanoid Robotics curriculum

**Quality Criteria:**
- Content completeness and accuracy (25 points)
- Structure and organization (15 points)
- Deployment success (10 points)

#### 2. Integrated RAG Chatbot (50 Points)

**Requirements:**
- Answer questions about textbook content
- Support user-selected text queries
- Provide context-aware assistance
- Embed seamlessly within published book
- Use specified technology stack

**Quality Criteria:**
- Accurate answer retrieval (20 points)
- User-selected text query support (15 points)
- Integration quality and UX (10 points)
- Response speed and reliability (5 points)

### Bonus Features (Up to 200 Additional Points)

#### 3. Reusable Intelligence via Claude Code (50 Points)

**Requirements:**
- Create Claude Code Subagents for specialized tasks
- Develop Agent Skills for reusable capabilities
- Document subagent/skill creation process
- Demonstrate reusability across project

**Evaluation Criteria:**
- Subagent design and documentation (20 points)
- Agent Skills development (20 points)
- Demonstrated reusability (10 points)

#### 4. Authentication System (50 Points)

**Requirements:**
- Implement Better-Auth for signup/signin
- Collect user background at signup:
  - Software development experience
  - Hardware/robotics background
  - Educational level
  - Learning goals
- Store profiles securely
- Enable authenticated features

**Evaluation Criteria:**
- Better-Auth implementation (20 points)
- User profile collection and storage (20 points)
- Security implementation (10 points)

#### 5. Content Personalization (50 Points)

**Requirements:**
- Add personalization button at chapter start
- Adapt content based on user profile:
  - Adjust technical depth
  - Provide relevant examples
  - Recommend prerequisites
  - Suggest advanced materials
- Maintain personalization preferences

**Evaluation Criteria:**
- Effective content adaptation (25 points)
- User experience quality (15 points)
- Technical implementation (10 points)

#### 6. Urdu Translation (50 Points)

**Requirements:**
- Implement translation toggle at chapter start
- Provide high-quality Urdu translations
- Maintain technical accuracy
- Ensure proper RTL text rendering
- Preserve formatting and code examples

**Evaluation Criteria:**
- Translation quality and accuracy (25 points)
- UI/UX for bilingual content (15 points)
- Technical term handling (10 points)

---

## V. Course Content Structure

### Physical AI & Humanoid Robotics Topics

The textbook must comprehensively cover:

#### 1. Foundations of Physical AI
- Introduction to Physical AI concepts
- Robotics fundamentals
- AI-robotics integration principles
- Historical context and future trends

#### 2. Hardware & Sensors
- Actuators and motors
- Sensors and perception systems
- Robot anatomy and kinematics
- Power systems and control electronics

#### 3. Software & AI Systems
- Robot Operating Systems (ROS)
- Computer vision for robotics
- Natural language processing for robots
- Machine learning for robotic systems
- Simulation and digital twins

#### 4. Control Systems
- Motion planning algorithms
- Navigation and localization
- Manipulation and grasping
- Real-time control systems

#### 5. Humanoid Robotics
- Bipedal locomotion
- Balance and stability
- Human-robot interaction
- Social robotics and emotion AI

#### 6. Applications & Future
- Industrial automation
- Healthcare robotics
- Service and companion robots
- Ethical considerations
- Future of human-AI-robot partnership

---

## VI. Development Workflow

### Phase 1: Planning & Setup

**Objectives:**
- Set up development environment
- Clone and configure Spec-Kit Plus
- Initialize Claude Code workspace
- Create project structure
- Define book outline

**Deliverables:**
- GitHub repository initialized
- Spec-Kit Plus configured
- Initial spec.md created
- Book structure documented

### Phase 2: Content Creation

**Objectives:**
- Write chapters using Claude Code
- Follow Spec-Kit Plus methodology
- Create diagrams and code examples
- Build progressive learning path
- Develop exercises and projects

**Deliverables:**
- All chapters written
- Code examples tested
- Diagrams and visualizations created
- Exercises designed
- Docusaurus site built

### Phase 3: RAG Chatbot Development

**Objectives:**
- Design chatbot architecture
- Set up FastAPI backend
- Configure Neon Postgres database
- Integrate Qdrant vector storage
- Implement OpenAI Agents SDK
- Test retrieval accuracy
- Embed chatbot in book

**Deliverables:**
- Backend API functional
- Database configured
- Vector embeddings created
- Chatbot integrated
- User-selected text query working

### Phase 4: Enhanced Features (Bonus)

**Objectives:**
- Implement Better-Auth authentication
- Create user profile system
- Build personalization engine
- Develop Urdu translation
- Create Claude Code Subagents
- Document Agent Skills

**Deliverables:**
- Authentication working
- User profiles stored
- Personalization functional
- Translation implemented
- Subagents/skills documented

### Phase 5: Testing & Deployment

**Objectives:**
- Comprehensive feature testing
- Performance optimization
- Security audit
- Documentation completion
- Deploy to GitHub Pages

**Deliverables:**
- All features tested
- Performance optimized
- Security validated
- Documentation complete
- Site deployed and accessible

---

## VII. Quality Standards

### Content Quality
- **Accuracy:** All technical content verified and correct
- **Clarity:** Clear explanations suitable for target audience
- **Completeness:** Comprehensive coverage of curriculum
- **Progression:** Logical flow from foundation to advanced
- **Relevance:** Real-world applications and examples

### Code Quality
- **Maintainability:** Clean, well-organized code
- **Documentation:** Clear comments and README files
- **Error Handling:** Comprehensive error management
- **Security:** Following best practices
- **Testing:** Validated functionality

### User Experience
- **Navigation:** Intuitive and consistent
- **Responsiveness:** Works across devices
- **Performance:** Fast load times
- **Accessibility:** WCAG compliance
- **Chatbot UX:** Smooth, helpful interactions

### AI Integration Quality
- **Effective Specs:** Clear specifications drive implementation
- **Reusable Components:** Skills and subagents properly designed
- **Documentation:** AI workflows clearly documented
- **Claude Code Usage:** Demonstrated effective collaboration

---

## VIII. Submission Requirements

### Mandatory Deliverables

1. **GitHub Repository URL**
   - All source code
   - Clear README with setup instructions
   - Documentation of project structure

2. **Live GitHub Pages URL**
   - Deployed and accessible textbook
   - All features functional
   - Proper configuration

3. **Technical Documentation**
   - AI tools usage documentation
   - RAG chatbot architecture
   - Setup and deployment guide

4. **Demo Materials**
   - Screenshots of key features
   - Video walkthrough (recommended)
   - Feature demonstration

### Bonus Feature Documentation

1. **Subagents & Skills Documentation**
   - Design rationale
   - Implementation details
   - Reusability demonstration

2. **Better-Auth Implementation Guide**
   - Setup instructions
   - Security considerations
   - User flow documentation

3. **Personalization Algorithm**
   - Logic explanation
   - User profile schema
   - Content adaptation strategy

4. **Translation Approach**
   - Translation methodology
   - Quality assurance process
   - RTL rendering implementation

---

## IX. Resources

### Essential Links
- **Spec-Kit Plus:** https://github.com/panaversity/spec-kit-plus/
- **Claude Code:** https://www.claude.com/product/claude-code
- **Better-Auth:** https://www.better-auth.com/

### Technical Documentation
- **Docusaurus:** https://docusaurus.io/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Neon Postgres:** https://neon.tech/
- **Qdrant:** https://qdrant.tech/
- **OpenAI SDK:** https://platform.openai.com/docs/

### Learning Resources
- **Physical AI Concepts:** Research papers and documentation
- **Humanoid Robotics:** Academic resources and case studies
- **ROS Documentation:** http://wiki.ros.org/
- **AI-Native Development:** Best practices and patterns

---

## X. Success Metrics

### Project Success Indicators

**Base Requirements (100 Points):**
- ✅ Textbook published and accessible via GitHub Pages
- ✅ All curriculum topics covered comprehensively
- ✅ RAG chatbot functional and embedded
- ✅ Chatbot answers questions accurately
- ✅ User-selected text query feature works
- ✅ Code examples tested and verified

**Bonus Features (Up to 200 Points):**
- ✅ Claude Code Subagents created and documented
- ✅ Users can sign up with Better-Auth
- ✅ Background information collected at signup
- ✅ Content personalizes based on user profile
- ✅ Urdu translation available and accurate
- ✅ All features integrated seamlessly

### Quality Indicators

**Technical Excellence:**
- Zero untested code in textbook
- All API integrations verified
- Security best practices followed
- Performance optimized

**Educational Effectiveness:**
- Clear learning progression
- Comprehensive topic coverage
- Practical examples and exercises
- Real-world application focus

**User Experience:**
- Intuitive navigation
- Responsive design
- Fast performance
- Accessible to all users

**AI Integration Quality:**
- Effective Specs: Clear specifications drive implementation
- Reusable Components: Skills and subagents properly designed
- Documentation: AI workflows clearly documented
- Claude Code Usage: Demonstrated effective collaboration

---

## XI. Governance

### Constitutional Authority

This constitution governs all aspects of textbook development:

1. **Design Principles** - Guide all architectural decisions
2. **Quality Standards** - Define acceptable quality levels
3. **Evaluation Criteria** - Determine project success
4. **Development Workflow** - Structure implementation process

### Amendment Process

**For Clarifications:**
- Update directly
- Document in version control
- Notify team

**For Major Changes:**
- Create proposal with rationale
- Team review and discussion
- Formal amendment if approved
- Update version number

---

## XII. Meta-Awareness: Self-Monitoring

### Avoiding Common Pitfalls

**Content Creation:**
- ❌ Don't create generic robotics content
- ✅ Create AI-native learning experiences
- ❌ Don't write without specifications
- ✅ Follow Spec-Kit Plus methodology
- ❌ Don't show code without context
- ✅ Explain WHY before showing HOW

**Technical Implementation:**
- ❌ Don't use localStorage (not supported)
- ✅ Use React state for client-side storage
- ❌ Don't implement without testing
- ✅ Test all features comprehensively
- ❌ Don't hardcode configurations
- ✅ Use environment variables

**AI Collaboration:**
- ❌ Don't use AI as just a code generator
- ✅ Collaborate with AI as development partner
- ❌ Don't accept AI output without validation
- ✅ Review and refine AI suggestions
- ❌ Don't create throwaway code
- ✅ Build reusable intelligence

### Self-Check Questions

Before finalizing any component, ask:

1. **Specification Quality**
   - Is there a clear spec before implementation?
   - Does spec articulate WHAT before HOW?
   - Can AI agent execute based on spec alone?

2. **Content Quality**
   - Is technical content accurate and verified?
   - Does progression make pedagogical sense?
   - Are examples relevant to Physical AI/robotics?

3. **Implementation Quality**
   - Is code clean and maintainable?
   - Are security best practices followed?
   - Is performance acceptable?

4. **User Experience Quality**
   - Is navigation intuitive?
   - Does chatbot provide helpful responses?
   - Are personalization/translation working?

5. **Reusability Quality**
   - Have we created reusable intelligence?
   - Are subagents/skills properly documented?
   - Can components be used in future projects?

---

**Version:** 1.0.0
**Last Updated:** November 29, 2025

This constitution activates reasoning mode through clear principles and frameworks. It defines WHAT to achieve and WHY it matters, while leaving HOW to contextual judgment and AI collaboration.