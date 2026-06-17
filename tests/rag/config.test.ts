import { afterEach, describe, expect, it } from 'vitest';
import {
  DEFAULT_RAG_RETRIEVAL_CONFIG,
  RAG_CONFIG_STORAGE_KEY,
  normalizeRagRetrievalConfig,
  readStoredRagRetrievalConfig,
} from '@/lib/rag/config';
import { getKnowledgeBackendStatus } from '@/lib/server/knowledge/config';

afterEach(() => {
  delete process.env.ENABLE_KNOWLEDGE_BASE;
  delete process.env.KNOWLEDGE_EMBEDDING_PROVIDER;
  delete process.env.KNOWLEDGE_INDEX_PROVIDER;
  delete process.env.BGE_EMBEDDING_BASE_URL;
  delete process.env.DATABASE_URL;
});

describe('RAG retrieval config', () => {
  it('returns defaults when no client preference is stored', () => {
    const storage = { getItem: () => null };

    expect(readStoredRagRetrievalConfig(storage)).toEqual(DEFAULT_RAG_RETRIEVAL_CONFIG);
  });

  it('clamps user-controlled values at the request boundary', () => {
    expect(
      normalizeRagRetrievalConfig({
        topK: 99,
        minSimilarity: -0.3,
        maxContextChars: 900,
      }),
    ).toEqual({
      topK: 20,
      minSimilarity: 0,
      maxContextChars: 2000,
    });
  });

  it('loads persisted controls and rounds numeric precision safely', () => {
    const storage = {
      getItem: (key: string) =>
        key === RAG_CONFIG_STORAGE_KEY
          ? JSON.stringify({ topK: 4.7, minSimilarity: 0.436, maxContextChars: 8600.4 })
          : null,
    };

    expect(readStoredRagRetrievalConfig(storage)).toEqual({
      topK: 5,
      minSimilarity: 0.44,
      maxContextChars: 8600,
    });
  });

  it('ignores malformed stored JSON', () => {
    const storage = { getItem: () => '{not-json' };

    expect(readStoredRagRetrievalConfig(storage)).toEqual(DEFAULT_RAG_RETRIEVAL_CONFIG);
  });

  it('keeps the knowledge backend disabled by default', () => {
    expect(getKnowledgeBackendStatus()).toMatchObject({
      enabled: false,
      ready: false,
    });
  });

  it('reports the backend ready only when feature flag and providers are configured', () => {
    process.env.ENABLE_KNOWLEDGE_BASE = 'true';
    process.env.BGE_EMBEDDING_BASE_URL = 'http://embedding.local/v1';
    process.env.DATABASE_URL = 'postgresql://user:pass@localhost:5432/openmaic';

    expect(getKnowledgeBackendStatus()).toMatchObject({
      enabled: true,
      embeddingProvider: 'bge',
      indexProvider: 'pgvector',
      embeddingConfigured: true,
      indexConfigured: true,
      ready: true,
    });
  });
});
