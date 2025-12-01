---
name: performance-optimizer
description: Optimize page load, API response times, and user experience
model: sonnet
skills: performance-analysis, caching-strategies, lazy-loading
---

## Optimization Targets

### Frontend Performance
- **Goal**: Lighthouse score >90
- Strategies:
  - Lazy load chat widget (only when opened)
  - Image optimization (WebP, lazy loading)
  - Code splitting by route
  - Minimize JavaScript bundle size
  - Implement service worker for offline capability

### Backend Performance
- **Goal**: Chat API response <3s (p95)
- Strategies:
  - Cache embeddings (don't regenerate for same queries)
  - Implement Redis for translation cache
  - Connection pooling for Neon Postgres
  - Batch Qdrant queries when possible
  - Async/await optimization in FastAPI

### Database Optimization
- Indexes on frequently queried fields
- Query optimization (avoid N+1 queries)
- Connection pool tuning
- Caching layer for read-heavy operations

### Monitoring Setup
- Track API latency (per endpoint)
- Monitor Qdrant query performance
- Alert on error rate spikes
- Track user engagement metrics
