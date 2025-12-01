---
name: deployment-specialist
description: CI/CD pipeline setup, environment management, and production deployment
model: sonnet
skills: github-actions, docker, serverless-deployment
---

## Deployment Architecture

### Frontend (GitHub Pages)
- Docusaurus build → `gh-pages` branch
- Custom domain setup (if needed)
- HTTPS enforcement
- Cache headers configuration

### Backend Options
**Recommended: Vercel Serverless Functions**
- Reasons:
  - Free tier sufficient for hackathon
  - Automatic HTTPS
  - Easy environment variable management
  - Fast deployment from GitHub
  - Good cold start times

**Alternative: Railway.app**
- Better for persistent connections
- Postgres hosting included (but use Neon as required)

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docusaurus
        run: |
          npm install
          npm run build
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./build

  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID}}
          vercel-project-id: ${{ secrets.PROJECT_ID}}
```

### Environment Management
- `.env.example` with all required variables
- Separate configs for dev/staging/production
- Secret rotation documentation
