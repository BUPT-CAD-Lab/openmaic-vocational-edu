import { createHash, randomUUID } from 'node:crypto';
import {
  ensureKnowledgeSchema,
  getKnowledgePool,
  getKnowledgeWorkspaceId,
  withKnowledgeTransaction,
} from './postgres';
import { embedWithBge, formatPgVector, getEmbeddingModel } from './embedding';
import { chunkKnowledgeText } from './chunker';
import type { RagEvidence, RagHit, RagSource } from '@/lib/types/rag';

const RETRIEVAL_LIMIT = 6;
const MAX_CONTEXT_CHARS = 12000;
const EMBEDDING_BATCH_SIZE = 32;

export interface KnowledgeDocument {
  id: string;
  fileName: string;
  mimeType: string;
  fileSize: number;
  status: string;
  embeddingModel: string;
  pageCount?: number;
  chunkCount: number;
  errorMessage?: string;
  createdAt: string;
  updatedAt: string;
}

type DocumentRow = {
  id: string;
  file_name: string;
  mime_type: string;
  file_size: string | number;
  status: string;
  embedding_model: string;
  page_count: number | null;
  chunk_count: number;
  error_message: string | null;
  created_at: Date | string;
  updated_at: Date | string;
};

function toDocument(row: DocumentRow): KnowledgeDocument {
  return {
    id: row.id,
    fileName: row.file_name,
    mimeType: row.mime_type,
    fileSize: Number(row.file_size),
    status: row.status,
    embeddingModel: row.embedding_model,
    pageCount: row.page_count ?? undefined,
    chunkCount: row.chunk_count,
    errorMessage: row.error_message ?? undefined,
    createdAt: new Date(row.created_at).toISOString(),
    updatedAt: new Date(row.updated_at).toISOString(),
  };
}

export async function listKnowledgeDocuments(): Promise<KnowledgeDocument[]> {
  await ensureKnowledgeSchema();
  const result = await getKnowledgePool().query<DocumentRow>(
    `SELECT id, file_name, mime_type, file_size, status, embedding_model, page_count,
            chunk_count, error_message, created_at, updated_at
       FROM knowledge_documents
      WHERE workspace_id = $1
      ORDER BY created_at DESC`,
    [getKnowledgeWorkspaceId()],
  );
  return result.rows.map(toDocument);
}

export async function findDocumentByHash(hash: string): Promise<KnowledgeDocument | undefined> {
  await ensureKnowledgeSchema();
  const result = await getKnowledgePool().query<DocumentRow>(
    `SELECT id, file_name, mime_type, file_size, status, embedding_model, page_count,
            chunk_count, error_message, created_at, updated_at
       FROM knowledge_documents
      WHERE workspace_id = $1 AND content_hash = $2
      LIMIT 1`,
    [getKnowledgeWorkspaceId(), hash],
  );
  return result.rows[0] ? toDocument(result.rows[0]) : undefined;
}

export function hashDocument(buffer: Buffer): string {
  return createHash('sha256').update(buffer).digest('hex');
}

export async function createKnowledgeDocument(input: {
  fileName: string;
  mimeType: string;
  buffer: Buffer;
  parsedText: string;
  pageCount?: number;
}): Promise<KnowledgeDocument> {
  const hash = hashDocument(input.buffer);
  const existing = await findDocumentByHash(hash);
  if (existing) return existing;

  const id = randomUUID();
  const workspaceId = getKnowledgeWorkspaceId();
  const model = getEmbeddingModel();
  await ensureKnowledgeSchema();
  await getKnowledgePool().query(
    `INSERT INTO knowledge_documents
      (id, workspace_id, file_name, mime_type, file_size, content_hash, pdf_data,
       parsed_text, page_count, status, embedding_model)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'embedding', $10)`,
    [
      id,
      workspaceId,
      input.fileName,
      input.mimeType,
      input.buffer.length,
      hash,
      input.buffer,
      input.parsedText,
      input.pageCount ?? null,
      model,
    ],
  );

  try {
    const chunks = chunkKnowledgeText(input.parsedText);
    if (chunks.length === 0) throw new Error('No searchable text was extracted from this PDF');
    const embeddings: number[][] = [];
    for (let start = 0; start < chunks.length; start += EMBEDDING_BATCH_SIZE) {
      embeddings.push(...(await embedWithBge(chunks.slice(start, start + EMBEDDING_BATCH_SIZE))));
    }
    await withKnowledgeTransaction(async (client) => {
      for (let index = 0; index < chunks.length; index++) {
        await client.query(
          `INSERT INTO knowledge_chunks
            (id, document_id, workspace_id, chunk_index, content, embedding)
           VALUES ($1, $2, $3, $4, $5, $6::vector)`,
          [randomUUID(), id, workspaceId, index, chunks[index], formatPgVector(embeddings[index])],
        );
      }
      await client.query(
        `UPDATE knowledge_documents
            SET status = 'ready', chunk_count = $2, updated_at = now()
          WHERE id = $1`,
        [id, chunks.length],
      );
    });
  } catch (error) {
    await getKnowledgePool().query(
      `UPDATE knowledge_documents
          SET status = 'failed', error_message = $2, updated_at = now()
        WHERE id = $1`,
      [id, error instanceof Error ? error.message : String(error)],
    );
    throw error;
  }

  const documents = await listKnowledgeDocuments();
  const document = documents.find((record) => record.id === id);
  if (!document) throw new Error('Stored knowledge document could not be read back');
  return document;
}

export async function deleteKnowledgeDocument(id: string): Promise<boolean> {
  await ensureKnowledgeSchema();
  const result = await getKnowledgePool().query(
    'DELETE FROM knowledge_documents WHERE id = $1 AND workspace_id = $2',
    [id, getKnowledgeWorkspaceId()],
  );
  return (result.rowCount || 0) > 0;
}

type RetrievedChunkRow = {
  document_id: string;
  file_name: string;
  chunk_index: number;
  content: string;
  similarity: number | string;
};

export interface RagSnapshot {
  id: string;
  context: string;
  hits: RagHit[];
  sources: RagSource[];
}

export async function retrieveKnowledgeSnapshot(query: string): Promise<RagSnapshot | undefined> {
  await ensureKnowledgeSchema();
  const ready = await getKnowledgePool().query<{ count: string }>(
    `SELECT count(*)::text AS count FROM knowledge_documents
      WHERE workspace_id = $1 AND status = 'ready'`,
    [getKnowledgeWorkspaceId()],
  );
  if (Number(ready.rows[0]?.count || 0) === 0) {
    throw new Error(
      'No indexed knowledge materials are ready. Upload a PDF in the knowledge base first.',
    );
  }

  const [embedding] = await embedWithBge([query]);
  const result = await getKnowledgePool().query<RetrievedChunkRow>(
    `SELECT c.document_id, d.file_name, c.chunk_index, c.content,
            1 - (c.embedding <=> $1::vector) AS similarity
       FROM knowledge_chunks c
       JOIN knowledge_documents d ON d.id = c.document_id
      WHERE c.workspace_id = $2 AND d.status = 'ready'
      ORDER BY c.embedding <=> $1::vector
      LIMIT $3`,
    [formatPgVector(embedding), getKnowledgeWorkspaceId(), RETRIEVAL_LIMIT],
  );
  if (result.rows.length === 0) return undefined;

  let usedChars = 0;
  const excerpts: string[] = [];
  const hits: RagHit[] = [];
  const sources = new Map<string, RagSource>();
  for (const row of result.rows) {
    const header = `[Source: ${row.file_name}, excerpt ${row.chunk_index + 1}]`;
    const remaining = MAX_CONTEXT_CHARS - usedChars - header.length - 2;
    if (remaining <= 0) break;
    const excerpt = row.content.slice(0, remaining);
    excerpts.push(`${header}\n${excerpt}`);
    usedChars += header.length + excerpt.length + 2;
    const similarity = Number(row.similarity);
    hits.push({
      documentId: row.document_id,
      documentName: row.file_name,
      chunkIndex: row.chunk_index,
      score: similarity,
      excerpt,
    });
    const source = sources.get(row.document_id);
    if (source) {
      source.score = Math.max(source.score, similarity);
      source.excerptCount++;
    } else {
      sources.set(row.document_id, {
        documentId: row.document_id,
        name: row.file_name,
        score: similarity,
        excerptCount: 1,
      });
    }
  }
  const context = `## Retrieved Knowledge Base Material\nUse these database-backed excerpts as factual grounding. Do not invent technical specifications or procedures that conflict with them.\n\n${excerpts.join('\n\n')}`;
  const id = randomUUID();
  const sourceValues = Array.from(sources.values());
  await getKnowledgePool().query(
    `INSERT INTO generation_rag_snapshots
      (id, workspace_id, query_text, retrieved_chunks, rendered_context)
     VALUES ($1, $2, $3, $4::jsonb, $5)`,
    [id, getKnowledgeWorkspaceId(), query, JSON.stringify(hits), context],
  );
  return { id, context, hits, sources: sourceValues };
}

export async function getRagSnapshotContext(id: string): Promise<string | undefined> {
  await ensureKnowledgeSchema();
  const result = await getKnowledgePool().query<{ rendered_context: string }>(
    `SELECT rendered_context FROM generation_rag_snapshots
      WHERE id = $1 AND workspace_id = $2 LIMIT 1`,
    [id, getKnowledgeWorkspaceId()],
  );
  return result.rows[0]?.rendered_context;
}

export async function getRagSnapshotEvidence(id: string): Promise<RagEvidence | undefined> {
  await ensureKnowledgeSchema();
  const result = await getKnowledgePool().query<{
    id: string;
    query_text: string;
    retrieved_chunks: unknown;
  }>(
    `SELECT id, query_text, retrieved_chunks
       FROM generation_rag_snapshots
      WHERE id = $1 AND workspace_id = $2 LIMIT 1`,
    [id, getKnowledgeWorkspaceId()],
  );
  const row = result.rows[0];
  if (!row) return undefined;

  const rawHits = Array.isArray(row.retrieved_chunks) ? row.retrieved_chunks : [];
  const hits = rawHits.filter(isRagHit);
  const sources = new Map<string, RagSource>();
  for (const hit of hits) {
    const source = sources.get(hit.documentId);
    if (source) {
      source.score = Math.max(source.score, hit.score);
      source.excerptCount++;
    } else {
      sources.set(hit.documentId, {
        documentId: hit.documentId,
        name: hit.documentName,
        score: hit.score,
        excerptCount: 1,
      });
    }
  }

  return {
    id: row.id,
    query: row.query_text,
    hits,
    sources: Array.from(sources.values()),
  };
}

function isRagHit(value: unknown): value is RagHit {
  if (!value || typeof value !== 'object') return false;
  const hit = value as Partial<RagHit>;
  return (
    typeof hit.documentId === 'string' &&
    typeof hit.documentName === 'string' &&
    typeof hit.chunkIndex === 'number' &&
    typeof hit.score === 'number' &&
    typeof hit.excerpt === 'string'
  );
}
