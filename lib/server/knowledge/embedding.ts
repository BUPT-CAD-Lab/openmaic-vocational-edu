export interface EmbeddingProvider {
  id: string;
  model: string;
  dimensions: number;
  embed(texts: string[]): Promise<number[][]>;
}

export interface ImageEmbeddingInput {
  id: string;
  mimeType?: string;
  data?: string;
  url?: string;
  description?: string;
}

export interface ImageEmbeddingProvider {
  id: string;
  model: string;
  dimensions: number;
  embedText(texts: string[]): Promise<number[][]>;
  embedImages(images: ImageEmbeddingInput[]): Promise<number[][]>;
}

export const DEFAULT_BGE_MODEL = 'BAAI/bge-m3';
export const DEFAULT_BGE_DIMENSIONS = 1024;
export const DEFAULT_JINA_CLIP_MODEL = 'jina-clip-v2';
export const DEFAULT_JINA_CLIP_DIMENSIONS = 1024;

function numberFromEnv(name: string, fallback: number): number {
  const raw = process.env[name]?.trim();
  if (!raw) return fallback;
  const value = Number(raw);
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

function getBgeConfig() {
  const baseUrl = process.env.BGE_EMBEDDING_BASE_URL?.trim();
  if (!baseUrl) {
    throw new Error('BGE_EMBEDDING_BASE_URL is not configured');
  }
  return {
    baseUrl: baseUrl.replace(/\/+$/, ''),
    apiKey: process.env.BGE_EMBEDDING_API_KEY?.trim(),
    model: process.env.BGE_EMBEDDING_MODEL?.trim() || DEFAULT_BGE_MODEL,
    dimensions: numberFromEnv('BGE_EMBEDDING_DIMENSIONS', DEFAULT_BGE_DIMENSIONS),
  };
}

export function isBgeEmbeddingConfigured(): boolean {
  return !!process.env.BGE_EMBEDDING_BASE_URL?.trim();
}

export function createBgeEmbeddingProvider(): EmbeddingProvider {
  const config = getBgeConfig();
  return {
    id: 'bge',
    model: config.model,
    dimensions: config.dimensions,
    async embed(texts: string[]): Promise<number[][]> {
      if (texts.length === 0) return [];
      const response = await fetch(`${config.baseUrl}/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : {}),
        },
        body: JSON.stringify({
          model: config.model,
          input: texts,
          encoding_format: 'float',
        }),
      });

      if (!response.ok) {
        const detail = await response.text().catch(() => '');
        throw new Error(
          `BGE embedding request failed (${response.status}): ${detail.slice(0, 240)}`,
        );
      }

      const payload = (await response.json()) as {
        data?: Array<{ index: number; embedding: number[] }>;
      };
      const embeddings = [...(payload.data || [])]
        .sort((left, right) => left.index - right.index)
        .map((item) => item.embedding);
      if (embeddings.length !== texts.length) {
        throw new Error('Embedding response did not include one vector per input text');
      }
      for (const embedding of embeddings) {
        if (embedding.length !== config.dimensions) {
          throw new Error(
            `Embedding dimension ${embedding.length} does not match provider dimension ${config.dimensions}`,
          );
        }
      }
      return embeddings;
    },
  };
}

export function getEmbeddingProvider(): EmbeddingProvider {
  const providerId = process.env.KNOWLEDGE_EMBEDDING_PROVIDER?.trim() || 'bge';
  if (providerId !== 'bge') {
    throw new Error(`Unknown embedding provider: ${providerId}`);
  }
  return createBgeEmbeddingProvider();
}

function getJinaClipConfig() {
  const baseUrl = process.env.JINA_CLIP_BASE_URL?.trim();
  if (!baseUrl) {
    throw new Error('JINA_CLIP_BASE_URL is not configured');
  }
  return {
    baseUrl: baseUrl.replace(/\/+$/, ''),
    apiKey: process.env.JINA_CLIP_API_KEY?.trim(),
    model: process.env.JINA_CLIP_MODEL?.trim() || DEFAULT_JINA_CLIP_MODEL,
    dimensions: numberFromEnv('JINA_CLIP_DIMENSIONS', DEFAULT_JINA_CLIP_DIMENSIONS),
  };
}

export function isJinaClipEmbeddingConfigured(): boolean {
  return !!process.env.JINA_CLIP_BASE_URL?.trim();
}

async function requestOpenAICompatibleEmbeddings(
  config: ReturnType<typeof getJinaClipConfig>,
  input: unknown[],
): Promise<number[][]> {
  if (input.length === 0) return [];
  const response = await fetch(`${config.baseUrl}/embeddings`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : {}),
    },
    body: JSON.stringify({
      model: config.model,
      input,
      encoding_format: 'float',
    }),
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(
      `Jina CLIP embedding request failed (${response.status}): ${detail.slice(0, 240)}`,
    );
  }

  const payload = (await response.json()) as {
    data?: Array<{ index: number; embedding: number[] }>;
  };
  const embeddings = [...(payload.data || [])]
    .sort((left, right) => left.index - right.index)
    .map((item) => item.embedding);
  if (embeddings.length !== input.length) {
    throw new Error('Image embedding response did not include one vector per input');
  }
  for (const embedding of embeddings) {
    if (embedding.length !== config.dimensions) {
      throw new Error(
        `Image embedding dimension ${embedding.length} does not match provider dimension ${config.dimensions}`,
      );
    }
  }
  return embeddings;
}

export function createJinaClipEmbeddingProvider(): ImageEmbeddingProvider {
  const config = getJinaClipConfig();
  return {
    id: 'jina-clip-v2',
    model: config.model,
    dimensions: config.dimensions,
    embedText(texts: string[]): Promise<number[][]> {
      return requestOpenAICompatibleEmbeddings(config, texts);
    },
    embedImages(images: ImageEmbeddingInput[]): Promise<number[][]> {
      return requestOpenAICompatibleEmbeddings(
        config,
        images.map((image) => {
          if (image.data) {
            return {
              type: 'image',
              image: image.data,
              mime_type: image.mimeType,
              text: image.description,
            };
          }
          if (image.url) {
            return {
              type: 'image_url',
              image_url: image.url,
              mime_type: image.mimeType,
              text: image.description,
            };
          }
          throw new Error(`Image asset ${image.id} does not include data or URL`);
        }),
      );
    },
  };
}

export function getImageEmbeddingProvider(): ImageEmbeddingProvider | undefined {
  const providerId = process.env.KNOWLEDGE_IMAGE_EMBEDDING_PROVIDER?.trim();
  if (!providerId || providerId === 'none') return undefined;
  if (providerId === 'jina-clip-v2') return createJinaClipEmbeddingProvider();
  throw new Error(`Unknown image embedding provider: ${providerId}`);
}

export async function embedWithBge(texts: string[]): Promise<number[][]> {
  return createBgeEmbeddingProvider().embed(texts);
}

export function getEmbeddingModel(): string {
  return getEmbeddingProvider().model;
}

export function formatPgVector(embedding: number[]): string {
  return `[${embedding.join(',')}]`;
}
