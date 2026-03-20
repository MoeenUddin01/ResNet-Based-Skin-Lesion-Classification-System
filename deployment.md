# Deployment Guide: Next.js + FastAPI (Free Tier Architecture)

This document outlines the architecture and exact steps used to deploy this machine learning application completely for free. You can use this guide as a template for future AI/ML web projects.

## Architecture Overview

Deploying heavy Deep Learning models (like a 220MB+ PyTorch ResNet) requires specific strategies because standard free hosting providers (like Vercel, Heroku, Render) impose strict memory and storage limits.

The solution is a **tri-partite split architecture**:

1. **Frontend (UI): Vercel** (Next.js App)
2. **Backend (API): Hugging Face Spaces** (FastAPI Docker Container)
3. **Model Storage: Hugging Face Hub** (Cloud Storage for large `.pth` files)

---

## Part 1: Backend Deployment (Hugging Face Spaces)

Hugging Face Spaces provides generous free CPU tiers specifically designed for Machine Learning inference.

### 1. Preparation
1. Ensure your backend has a `Dockerfile` exposing port `7860` (the HF Spaces default).
2. Create a clean `requirements.txt` containing **only** the dependencies needed for inference (e.g., `fastapi`, `uvicorn`, `torch`, `torchvision`, `Pillow`). Exclude training tools to keep the image small.
3. Ensure CORS is configured in your API to allow requests from your future Vercel frontend.

```python
# app/middleware.py
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

### 2. Code Adjustments
Your `README.md` must include YAML frontmatter at the top so the Space parses it as a Docker app:
```yaml
---
title: My Cool API
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

### 3. Deployment
1. Create a new **Blank Space** on Hugging Face (choose Docker as the SDK).
2. Add the space as a git remote in your local repository:
   ```bash
   git remote add hf-space https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
   ```
3. Authenticate with Hugging Face (you can generate a token in Settings -> Access Tokens):
   ```bash
   huggingface-cli login
   # Or manually inject it: git config --global credential.helper store
   ```
4. Push your code:
   ```bash
   git push hf-space main
   ```

---

## Part 2: Model Storage (Hugging Face Hub)

GitHub will reject files over 100MB, and Vercel limits Serverless Functions to 50MB. Heavy weights must be stored independently.

### 1. Create Model Repo
1. Go to Hugging Face and create a new **Model** (e.g., `YOUR_USERNAME/my-model-weights`).

### 2. Upload Weights
Use the CLI to upload your heavy `.pth` file directly into the model repository:
```bash
huggingface-cli upload YOUR_USERNAME/my-model-weights path/to/local/model.pth model.pth
```

### 3. Connect API to Model
Modify your inference code to download the weights on startup:
```python
from huggingface_hub import hf_hub_download

repo_id = os.getenv("HF_MODEL_REPO")
model_path = hf_hub_download(repo_id=repo_id, filename="model.pth")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
```
*Note: Go to your HF Space Settings -> Repository Secrets and add `HF_MODEL_REPO` as a secret pointing to your model repo from Step 1.*

---

## Part 3: Frontend Deployment (Vercel)

Vercel is the optimal host for Next.js applications, offering edge caching and seamless deployment.

### 1. Configuration
Modify your frontend's `next.config.ts` so it dynamically routes API calls.
You should define a `NEXT_PUBLIC_API_URL` environment variable.

```typescript
// next.config.ts
const nextConfig = {
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [
      {
        source: "/api/v1/:path*",
        destination: `${backendUrl}/api/v1/:path*`,
      },
    ];
  },
};
```

### 2. Deployment options
**Option A: Vercel Dashboard (Recommended for Monorepos)**
If your Next.js app is inside a subfolder (like `frontend/`), the easiest way is the web dashboard:
1. Go to Vercel -> Add New Project -> Import GitHub Repo.
2. In the "Root Directory" setting, click Edit and select `frontend`.
3. Add the `NEXT_PUBLIC_API_URL` variable pointing to your Hugging Face Space URL (e.g., `https://your-username-space-name.hf.space`).
4. Click Deploy.

**Option B: Vercel CLI**
If you prefer the command line:
1. Navigate into the frontend folder: `cd frontend`
2. Run the deployment sequence, explicitly accepting defaults and injecting the variable:
```bash
npx vercel link --yes
npx vercel pull --yes --environment=production
npx vercel --prod --yes --env NEXT_PUBLIC_API_URL=https://YOUR-HF-SPACE.hf.space --build-env NEXT_PUBLIC_API_URL=https://YOUR-HF-SPACE.hf.space
```

### 3. Error Handling Note
When separating frontend and backend, always ensure your frontend `fetch` specifically handles non-JSON responses (like 500 HTML error pages), otherwise it will crash with `Unexpected token 'I', "Internal S"... is not valid JSON`.
Always read the response as text first in your `try/catch` block, then parse as JSON if applicable.
