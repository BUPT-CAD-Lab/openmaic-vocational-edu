# Multimodal RAG Deployment Guide

This guide describes how to deploy the local multimodal RAG stack used by the
OpenMAIC knowledge-base workflow on this branch.

The feature supports:

- document ingestion from the knowledge-base page
- text chunk embedding and pgvector indexing
- image extraction from document artifacts
- optional image-vector embedding with Jina CLIP v2
- retrieve-then-confirm evidence selection before generation
- classroom-side visualization of the text and image evidence used for generation

## Architecture

```text
Document upload
  -> DocumentExtractorProvider
  -> DocumentArtifact
  -> text chunks + image chunks
  -> BGE text embeddings + optional Jina CLIP image embeddings
  -> PostgreSQL / pgvector
  -> retrieval snapshot
  -> user-confirmed evidence
  -> generation prompts
  -> classroom evidence sidebar
```

## Required Services

### 1. PostgreSQL with pgvector

The branch includes a `knowledge` compose profile for local pgvector.

```bash
docker compose --profile knowledge up -d postgres
```

Check the database container:

```bash
docker compose --profile knowledge ps postgres
```

Default connection string:

```bash
postgresql://openmaic:openmaic@localhost:5433/openmaic
```

The application creates the knowledge tables and pgvector extension on first use.

### 2. Text Embedding Service

The text embedding provider expects an OpenAI-compatible embeddings endpoint:

```text
POST <BGE_EMBEDDING_BASE_URL>/embeddings
```

Recommended model:

```text
BAAI/bge-m3
```

Example environment values:

```bash
BGE_EMBEDDING_BASE_URL=http://localhost:8200/v1
BGE_EMBEDDING_MODEL=BAAI/bge-m3
BGE_EMBEDDING_DIMENSIONS=1024
```

You may use any server that exposes an OpenAI-compatible `/v1/embeddings` API.
For vLLM versions that support embedding tasks, a typical command is:

```bash
vllm serve BAAI/bge-m3 \
  --task embed \
  --host 0.0.0.0 \
  --port 8200
```

If your vLLM version does not support that flag set, keep the OpenMAIC env vars
the same and use another OpenAI-compatible embedding server.

### 3. MinerU Document Extraction

MinerU is recommended when you want richer PDF/DOCX/PPTX extraction, including
document images, layout, tables, and formulas.

If you use a separately maintained MinerU deployment directory, start its API
service first. Example:

```bash
cd mineru-deploy
docker compose -f compose.yaml --profile api up -d
```

Check health:

```bash
curl -f http://localhost:7777/health
```

Configure OpenMAIC to use that service:

```bash
PDF_MINERU_BASE_URL=http://localhost:7777
KNOWLEDGE_DOCUMENT_EXTRACTOR_PROVIDER=mineru
```

If MinerU is not configured, keep `KNOWLEDGE_DOCUMENT_EXTRACTOR_PROVIDER`
unset. In that mode the knowledge-base upload still works for text-based
formats supported by the local extractors, but multimodal document extraction is
limited.

### 4. Jina CLIP v2 Image Embedding Service

Image-vector retrieval is optional. Enable it when you want uploaded document
images to be embedded and retrieved semantically alongside text.

Download the model:

```bash
huggingface-cli download jinaai/jina-clip-v2 \
  --local-dir llms/jina-clip-v2
```

Install Python dependencies in your preferred environment:

```bash
pip install fastapi uvicorn pillow sentence-transformers torch
```

Start the local OpenAI-compatible helper server included in this branch:

```bash
python scripts/serve-jina-clip-v2.py \
  --model-path llms/jina-clip-v2 \
  --host 0.0.0.0 \
  --port 8300
```

Check health:

```bash
curl -f http://localhost:8300/health
```

Expected response includes:

```json
{
  "status": "healthy",
  "model": "jina-clip-v2"
}
```

The helper script disables optional xFormers attention flags when xFormers is
not installed, which avoids the common runtime error:

```text
Can't use xattn without xformers
```

## OpenMAIC Environment

Create or update `.env.local`:

```bash
ENABLE_KNOWLEDGE_BASE=true

KNOWLEDGE_INDEX_PROVIDER=pgvector
DATABASE_URL=postgresql://openmaic:openmaic@localhost:5433/openmaic
KNOWLEDGE_WORKSPACE_ID=demo-default

KNOWLEDGE_EMBEDDING_PROVIDER=bge
BGE_EMBEDDING_BASE_URL=http://localhost:8200/v1
BGE_EMBEDDING_MODEL=BAAI/bge-m3
BGE_EMBEDDING_DIMENSIONS=1024

KNOWLEDGE_IMAGE_EMBEDDING_PROVIDER=jina-clip-v2
JINA_CLIP_BASE_URL=http://localhost:8300/v1
JINA_CLIP_MODEL=jina-clip-v2
JINA_CLIP_DIMENSIONS=1024

PDF_MINERU_BASE_URL=http://localhost:7777
KNOWLEDGE_DOCUMENT_EXTRACTOR_PROVIDER=mineru
```

If you only want text RAG, disable image-vector retrieval:

```bash
KNOWLEDGE_IMAGE_EMBEDDING_PROVIDER=none
```

After changing `.env.local`, restart the Next.js dev server.

## Start OpenMAIC

```bash
pnpm install
pnpm dev
```

Open:

```text
http://localhost:3000
```

## Manual Test Flow

### 1. Verify backend readiness

Open the homepage and confirm the local knowledge toggle is available. If it is
not visible, call:

```bash
curl -s http://localhost:3000/api/server-providers
```

The `knowledgeBase` payload should show:

```json
{
  "enabled": true,
  "ready": true,
  "embeddingConfigured": true,
  "indexConfigured": true
}
```

When image RAG is enabled, `imageEmbeddingConfigured` should also be `true`.

### 2. Upload knowledge material

Open:

```text
http://localhost:3000/knowledge-base
```

Upload one supported document, for example:

- PDF
- DOCX
- PPTX
- TXT
- Markdown

For multimodal testing, prefer a PDF/DOCX/PPTX that contains meaningful images.

Expected result:

- the document status becomes ready
- `chunk_count` includes text chunks and image chunks
- no database UTF-8 errors are reported
- no Jina CLIP embedding errors are reported

### 3. Generate with local knowledge

Go back to:

```text
http://localhost:3000
```

Enable local knowledge in the generation toolbar, enter a course-generation
request, and start generation.

The generation preview should:

- retrieve candidate evidence from the local knowledge base
- show text evidence
- show image evidence when image-vector retrieval finds matches
- allow manual selection of the text and image snippets used for generation

Confirm the selected evidence before generation continues.

### 4. Inspect generated classroom evidence

After generation opens the classroom page:

```text
http://localhost:3000/classroom/<classroom-id>
```

The classroom should show a right-side evidence dock when the stage has a
`ragSnapshotId`.

The dock displays:

- retrieved text snippets
- retrieved image thumbnails
- source document names
- page/location metadata when available
- similarity scores

## Troubleshooting

### Knowledge toggle is not visible

Check:

```bash
curl -s http://localhost:3000/api/server-providers
```

Common causes:

- `ENABLE_KNOWLEDGE_BASE` is not `true`
- `DATABASE_URL` is missing
- the BGE embedding service is not reachable
- the Next.js dev server was not restarted after changing `.env.local`

### DOCX/PPTX upload says MinerU is required

Self-hosted MinerU needs a configured base URL:

```bash
PDF_MINERU_BASE_URL=http://localhost:7777
KNOWLEDGE_DOCUMENT_EXTRACTOR_PROVIDER=mineru
```

Then restart OpenMAIC.

### Image evidence does not appear

Check:

```bash
curl -f http://localhost:8300/health
```

Then verify:

```bash
KNOWLEDGE_IMAGE_EMBEDDING_PROVIDER=jina-clip-v2
JINA_CLIP_BASE_URL=http://localhost:8300/v1
```

Also make sure the uploaded document actually produced image assets. Documents
without extracted images can still produce text evidence, but will not produce
image evidence.

### `pnpm check` scans local course dumps

Do not keep large downloaded course-material dumps inside the repo root when
running full-repo Prettier checks. For example, a local `20260608/` directory may
contain third-party WebGL assets that are not part of this branch and are not
valid Prettier input.

Before pushing, confirm the branch does not include local dumps:

```bash
git diff --name-only 5a15a2c..HEAD | rg '^20260608/'
```

No output means the local data directory is not included in the branch.

