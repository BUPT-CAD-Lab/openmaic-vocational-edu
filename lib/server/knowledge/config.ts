import { isBgeEmbeddingConfigured, isJinaClipEmbeddingConfigured } from './embedding';
import { isPgVectorConfigured } from './postgres';

export function isKnowledgeBaseFeatureEnabled(): boolean {
  return /^(true|1|yes|on)$/i.test(process.env.ENABLE_KNOWLEDGE_BASE?.trim() || '');
}

export function getKnowledgeBackendStatus(): {
  enabled: boolean;
  embeddingProvider: string;
  imageEmbeddingProvider?: string;
  indexProvider: string;
  embeddingConfigured: boolean;
  imageEmbeddingConfigured: boolean;
  indexConfigured: boolean;
  ready: boolean;
  reason?: string;
} {
  const enabled = isKnowledgeBaseFeatureEnabled();
  const embeddingProvider = process.env.KNOWLEDGE_EMBEDDING_PROVIDER?.trim() || 'bge';
  const imageEmbeddingProvider = process.env.KNOWLEDGE_IMAGE_EMBEDDING_PROVIDER?.trim() || 'none';
  const indexProvider = process.env.KNOWLEDGE_INDEX_PROVIDER?.trim() || 'pgvector';
  const embeddingConfigured = embeddingProvider === 'bge' && isBgeEmbeddingConfigured();
  const imageEmbeddingConfigured =
    imageEmbeddingProvider === 'none' ||
    imageEmbeddingProvider === '' ||
    (imageEmbeddingProvider === 'jina-clip-v2' && isJinaClipEmbeddingConfigured());
  const indexConfigured = indexProvider === 'pgvector' && isPgVectorConfigured();
  const ready = enabled && embeddingConfigured && indexConfigured;
  let reason: string | undefined;
  if (!enabled) reason = 'Knowledge base is disabled by ENABLE_KNOWLEDGE_BASE';
  else if (!embeddingConfigured) reason = 'Embedding provider is not configured';
  else if (!indexConfigured) reason = 'Knowledge index provider is not configured';
  return {
    enabled,
    embeddingProvider,
    imageEmbeddingProvider,
    indexProvider,
    embeddingConfigured,
    imageEmbeddingConfigured,
    indexConfigured,
    ready,
    reason,
  };
}

export function assertKnowledgeBackendReady(): void {
  const status = getKnowledgeBackendStatus();
  if (!status.ready) {
    throw new Error(status.reason || 'Knowledge backend is not configured');
  }
}
