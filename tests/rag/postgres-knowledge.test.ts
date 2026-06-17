import { afterEach, describe, expect, it, vi } from 'vitest';
import { chunkKnowledgeText } from '@/lib/server/knowledge/chunker';
import {
  DEFAULT_BGE_MODEL,
  DEFAULT_JINA_CLIP_MODEL,
  createBgeEmbeddingProvider,
  createJinaClipEmbeddingProvider,
  embedWithBge,
  formatPgVector,
  getImageEmbeddingProvider,
} from '@/lib/server/knowledge/embedding';

afterEach(() => {
  vi.unstubAllGlobals();
  delete process.env.KNOWLEDGE_EMBEDDING_PROVIDER;
  delete process.env.BGE_EMBEDDING_BASE_URL;
  delete process.env.BGE_EMBEDDING_API_KEY;
  delete process.env.BGE_EMBEDDING_MODEL;
  delete process.env.BGE_EMBEDDING_DIMENSIONS;
  delete process.env.KNOWLEDGE_IMAGE_EMBEDDING_PROVIDER;
  delete process.env.JINA_CLIP_BASE_URL;
  delete process.env.JINA_CLIP_API_KEY;
  delete process.env.JINA_CLIP_MODEL;
  delete process.env.JINA_CLIP_DIMENSIONS;
});

describe('PostgreSQL knowledge base helpers', () => {
  it('retains text beyond the original direct-PDF prompt limit when chunking', () => {
    const text = `${'U660E 变速箱阀体油压测试步骤。'.repeat(4000)}\n结束标记`;
    const chunks = chunkKnowledgeText(text);

    expect(text.length).toBeGreaterThan(50000);
    expect(chunks.length).toBeGreaterThan(40);
    expect(chunks.at(-1)).toContain('结束标记');
  });

  it('serializes embedding values for pgvector parameters', () => {
    expect(formatPgVector([0.125, -0.5, 1])).toBe('[0.125,-0.5,1]');
  });

  it('requests 1024-dimensional BGE embeddings from an OpenAI-compatible endpoint', async () => {
    process.env.BGE_EMBEDDING_BASE_URL = 'http://embedding.local/v1';
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [{ index: 0, embedding: Array.from({ length: 1024 }, () => 0.1) }],
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await embedWithBge(['阀体油压测试']);

    expect(result[0]).toHaveLength(1024);
    expect(fetchMock).toHaveBeenCalledWith(
      'http://embedding.local/v1/embeddings',
      expect.objectContaining({
        body: JSON.stringify({
          model: DEFAULT_BGE_MODEL,
          input: ['阀体油压测试'],
          encoding_format: 'float',
        }),
      }),
    );
  });

  it('takes the embedding dimensions from the selected provider config', async () => {
    process.env.BGE_EMBEDDING_BASE_URL = 'http://embedding.local/v1';
    process.env.BGE_EMBEDDING_DIMENSIONS = '3';
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            data: [{ index: 0, embedding: [0.1, 0.2, 0.3] }],
          }),
          { status: 200 },
        ),
      ),
    );

    const provider = createBgeEmbeddingProvider();
    const result = await provider.embed(['短文本']);

    expect(provider.dimensions).toBe(3);
    expect(result).toEqual([[0.1, 0.2, 0.3]]);
  });

  it('keeps image embedding optional until a provider is configured', () => {
    expect(getImageEmbeddingProvider()).toBeUndefined();

    process.env.KNOWLEDGE_IMAGE_EMBEDDING_PROVIDER = 'none';
    expect(getImageEmbeddingProvider()).toBeUndefined();
  });

  it('uses jina-clip-v2 for configured cross-modal image embeddings', async () => {
    process.env.KNOWLEDGE_IMAGE_EMBEDDING_PROVIDER = 'jina-clip-v2';
    process.env.JINA_CLIP_BASE_URL = 'http://jina-clip.local/v1';
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [{ index: 0, embedding: Array.from({ length: 1024 }, () => 0.2) }],
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const provider = createJinaClipEmbeddingProvider();
    const result = await provider.embedImages([
      {
        id: 'img-1',
        mimeType: 'image/png',
        data: 'data:image/png;base64,abc',
        description: '液压泵结构图',
      },
    ]);

    expect(provider.model).toBe(DEFAULT_JINA_CLIP_MODEL);
    expect(result[0]).toHaveLength(1024);
    expect(fetchMock).toHaveBeenCalledWith(
      'http://jina-clip.local/v1/embeddings',
      expect.objectContaining({
        body: JSON.stringify({
          model: DEFAULT_JINA_CLIP_MODEL,
          input: [
            {
              type: 'image',
              image: 'data:image/png;base64,abc',
              mime_type: 'image/png',
              text: '液压泵结构图',
            },
          ],
          encoding_format: 'float',
        }),
      }),
    );
  });
});
