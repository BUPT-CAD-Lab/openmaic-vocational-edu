import { Pool, type PoolClient } from 'pg';
import type { RagHit, RagRetrievalConfig } from '@/lib/types/rag';
import { formatPgVector } from './embedding';
import type { KnowledgeUnitCitation, KnowledgeUnitModality } from './document-units';

const DEFAULT_WORKSPACE_ID = 'demo-default';

type GlobalWithPool = typeof globalThis & {
  __openmaicKnowledgePool?: Pool;
  __openmaicKnowledgeSchema?: Map<string, Promise<void>>;
};

const globalForPool = globalThis as GlobalWithPool;

export interface StoredKnowledgeDocumentInput {
  id: string;
  workspaceId: string;
  fileName: string;
  mimeType: string;
  fileSize: number;
  contentHash: string;
  data: Buffer;
  parsedText: string;
  pageCount?: number;
  embeddingModel: string;
}

export interface IndexedKnowledgeChunk {
  id: string;
  documentId: string;
  workspaceId: string;
  chunkIndex: number;
  content: string;
  embedding: number[];
  modality?: KnowledgeUnitModality;
  citation?: KnowledgeUnitCitation;
  assetIds?: string[];
}

export interface IndexedKnowledgeImageChunk {
  id: string;
  documentId: string;
  workspaceId: string;
  chunkIndex: number;
  assetId: string;
  content: string;
  imageData?: string;
  imageMimeType?: string;
  embedding: number[];
  citation?: KnowledgeUnitCitation;
}

export type DocumentRow = {
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

export type RetrievedChunkRow = {
  document_id: string;
  file_name: string;
  chunk_index: number;
  content: string;
  similarity: number | string;
  modality: string | null;
  citation: unknown;
  asset_ids: unknown;
};

export type RetrievedImageChunkRow = {
  id: string;
  document_id: string;
  file_name: string;
  chunk_index: number;
  asset_id: string;
  content: string;
  image_data: string | null;
  image_mime_type: string | null;
  similarity: number | string;
  citation: unknown;
};

export interface KnowledgeIndexProvider {
  id: string;
  vectorDimensions: number;
  imageVectorDimensions?: number;
  workspaceId: string;
  ensure(): Promise<void>;
  listDocuments(): Promise<DocumentRow[]>;
  findDocumentByHash(hash: string): Promise<DocumentRow | undefined>;
  insertDocument(input: StoredKnowledgeDocumentInput): Promise<void>;
  upsertChunks(chunks: IndexedKnowledgeChunk[]): Promise<void>;
  upsertImageChunks(chunks: IndexedKnowledgeImageChunk[]): Promise<void>;
  markDocumentReady(documentId: string, chunkCount: number): Promise<void>;
  markDocumentFailed(documentId: string, error: string): Promise<void>;
  deleteDocument(documentId: string): Promise<boolean>;
  countReadyDocuments(): Promise<number>;
  queryChunks(embedding: number[], config: RagRetrievalConfig): Promise<RetrievedChunkRow[]>;
  queryImageChunks(
    embedding: number[],
    config: RagRetrievalConfig,
  ): Promise<RetrievedImageChunkRow[]>;
  insertSnapshot(input: {
    id: string;
    query: string;
    config: RagRetrievalConfig;
    hits: RagHit[];
    context: string;
    selectionConfirmed: boolean;
  }): Promise<void>;
  getSnapshotContext(id: string): Promise<string | undefined>;
  getSnapshotEvidenceRow(id: string): Promise<
    | {
        id: string;
        query_text: string;
        retrieval_config: unknown;
        selection_confirmed: boolean;
        retrieved_chunks: unknown;
      }
    | undefined
  >;
  updateSnapshotSelection(id: string, hits: RagHit[], context: string): Promise<void>;
}

export function getKnowledgeWorkspaceId(): string {
  return process.env.KNOWLEDGE_WORKSPACE_ID?.trim() || DEFAULT_WORKSPACE_ID;
}

function requireDatabaseUrl(): string {
  const databaseUrl = process.env.DATABASE_URL?.trim();
  if (!databaseUrl) {
    throw new Error('DATABASE_URL is not configured for the PostgreSQL knowledge base');
  }
  return databaseUrl;
}

export function isPgVectorConfigured(): boolean {
  return !!process.env.DATABASE_URL?.trim();
}

function getKnowledgePool(): Pool {
  if (!globalForPool.__openmaicKnowledgePool) {
    globalForPool.__openmaicKnowledgePool = new Pool({
      connectionString: requireDatabaseUrl(),
      max: 8,
    });
  }
  return globalForPool.__openmaicKnowledgePool;
}

async function ensureKnowledgeSchema(
  vectorDimensions: number,
  imageVectorDimensions?: number,
): Promise<void> {
  if (!globalForPool.__openmaicKnowledgeSchema) {
    globalForPool.__openmaicKnowledgeSchema = new Map();
  }
  const cacheKey = `${vectorDimensions}:${imageVectorDimensions ?? 0}`;
  const cached = globalForPool.__openmaicKnowledgeSchema.get(cacheKey);
  if (cached) return cached;

  const promise = initializeKnowledgeSchema(vectorDimensions, imageVectorDimensions).catch(
    (error) => {
      globalForPool.__openmaicKnowledgeSchema?.delete(cacheKey);
      throw error;
    },
  );
  globalForPool.__openmaicKnowledgeSchema.set(cacheKey, promise);
  return promise;
}

async function initializeKnowledgeSchema(
  vectorDimensions: number,
  imageVectorDimensions?: number,
): Promise<void> {
  const client = await getKnowledgePool().connect();
  try {
    await client.query('CREATE EXTENSION IF NOT EXISTS vector');
    await client.query(`
      CREATE TABLE IF NOT EXISTS knowledge_documents (
        id text PRIMARY KEY,
        workspace_id text NOT NULL,
        file_name text NOT NULL,
        mime_type text NOT NULL,
        file_size bigint NOT NULL,
        content_hash text NOT NULL,
        document_data bytea NOT NULL,
        parsed_text text,
        page_count integer,
        chunk_count integer NOT NULL DEFAULT 0,
        status text NOT NULL,
        embedding_model text NOT NULL,
        error_message text,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE (workspace_id, content_hash)
      )
    `);
    await client.query(`
      ALTER TABLE knowledge_documents
      ADD COLUMN IF NOT EXISTS document_data bytea
    `);
    await client.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
            FROM information_schema.columns
           WHERE table_schema = 'public'
             AND table_name = 'knowledge_documents'
             AND column_name = 'pdf_data'
        ) THEN
          UPDATE knowledge_documents
             SET document_data = COALESCE(document_data, pdf_data)
           WHERE document_data IS NULL;

          ALTER TABLE knowledge_documents
          ALTER COLUMN pdf_data DROP NOT NULL;
        END IF;
      END $$;
    `);
    await client.query(`
      CREATE TABLE IF NOT EXISTS knowledge_chunks (
        id text PRIMARY KEY,
        document_id text NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
        workspace_id text NOT NULL,
        chunk_index integer NOT NULL,
        content text NOT NULL,
        modality text NOT NULL DEFAULT 'text',
        citation jsonb NOT NULL DEFAULT '{}'::jsonb,
        asset_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
        embedding vector(${vectorDimensions}) NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE (document_id, chunk_index)
      )
    `);
    await client.query(`
      ALTER TABLE knowledge_chunks
      ADD COLUMN IF NOT EXISTS modality text NOT NULL DEFAULT 'text'
    `);
    await client.query(`
      ALTER TABLE knowledge_chunks
      ADD COLUMN IF NOT EXISTS citation jsonb NOT NULL DEFAULT '{}'::jsonb
    `);
    await client.query(`
      ALTER TABLE knowledge_chunks
      ADD COLUMN IF NOT EXISTS asset_ids jsonb NOT NULL DEFAULT '[]'::jsonb
    `);
    await client.query(`
      CREATE TABLE IF NOT EXISTS generation_rag_snapshots (
        id text PRIMARY KEY,
        workspace_id text NOT NULL,
        query_text text NOT NULL,
        retrieval_config jsonb NOT NULL DEFAULT '{}'::jsonb,
        selection_confirmed boolean NOT NULL DEFAULT true,
        retrieved_chunks jsonb NOT NULL,
        rendered_context text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now()
      )
    `);
    await client.query(`
      CREATE INDEX IF NOT EXISTS knowledge_chunks_workspace_document_idx
      ON knowledge_chunks (workspace_id, document_id)
    `);
    await client.query(`
      CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_hnsw_idx
      ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)
    `);
    if (imageVectorDimensions) {
      await client.query(`
        CREATE TABLE IF NOT EXISTS knowledge_image_chunks (
          id text PRIMARY KEY,
          document_id text NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
          workspace_id text NOT NULL,
          chunk_index integer NOT NULL,
          asset_id text NOT NULL,
          content text NOT NULL,
          image_data text,
          image_mime_type text,
          citation jsonb NOT NULL DEFAULT '{}'::jsonb,
          embedding vector(${imageVectorDimensions}) NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          UNIQUE (document_id, chunk_index)
        )
      `);
      await client.query(`
        CREATE INDEX IF NOT EXISTS knowledge_image_chunks_workspace_document_idx
        ON knowledge_image_chunks (workspace_id, document_id)
      `);
      await client.query(`
        CREATE INDEX IF NOT EXISTS knowledge_image_chunks_embedding_hnsw_idx
        ON knowledge_image_chunks USING hnsw (embedding vector_cosine_ops)
      `);
    }
  } finally {
    client.release();
  }
}

async function withKnowledgeTransaction<T>(
  vectorDimensions: number,
  imageVectorDimensions: number | undefined,
  operation: (client: PoolClient) => Promise<T>,
): Promise<T> {
  await ensureKnowledgeSchema(vectorDimensions, imageVectorDimensions);
  const client = await getKnowledgePool().connect();
  try {
    await client.query('BEGIN');
    const result = await operation(client);
    await client.query('COMMIT');
    return result;
  } catch (error) {
    await client.query('ROLLBACK');
    throw error;
  } finally {
    client.release();
  }
}

export class PgVectorKnowledgeIndexProvider implements KnowledgeIndexProvider {
  readonly id = 'pgvector';
  readonly workspaceId = getKnowledgeWorkspaceId();

  constructor(
    readonly vectorDimensions: number,
    readonly imageVectorDimensions?: number,
  ) {}

  async ensure(): Promise<void> {
    await ensureKnowledgeSchema(this.vectorDimensions, this.imageVectorDimensions);
  }

  async listDocuments(): Promise<DocumentRow[]> {
    await this.ensure();
    const result = await getKnowledgePool().query<DocumentRow>(
      `SELECT id, file_name, mime_type, file_size, status, embedding_model, page_count,
              chunk_count, error_message, created_at, updated_at
         FROM knowledge_documents
        WHERE workspace_id = $1
        ORDER BY created_at DESC`,
      [this.workspaceId],
    );
    return result.rows;
  }

  async findDocumentByHash(hash: string): Promise<DocumentRow | undefined> {
    await this.ensure();
    const result = await getKnowledgePool().query<DocumentRow>(
      `SELECT id, file_name, mime_type, file_size, status, embedding_model, page_count,
              chunk_count, error_message, created_at, updated_at
         FROM knowledge_documents
        WHERE workspace_id = $1 AND content_hash = $2
        LIMIT 1`,
      [this.workspaceId, hash],
    );
    return result.rows[0];
  }

  async insertDocument(input: StoredKnowledgeDocumentInput): Promise<void> {
    await this.ensure();
    await getKnowledgePool().query(
      `INSERT INTO knowledge_documents
        (id, workspace_id, file_name, mime_type, file_size, content_hash, document_data,
         parsed_text, page_count, status, embedding_model)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'embedding', $10)`,
      [
        input.id,
        input.workspaceId,
        input.fileName,
        input.mimeType,
        input.fileSize,
        input.contentHash,
        input.data,
        input.parsedText,
        input.pageCount ?? null,
        input.embeddingModel,
      ],
    );
  }

  async upsertChunks(chunks: IndexedKnowledgeChunk[]): Promise<void> {
    await withKnowledgeTransaction(
      this.vectorDimensions,
      this.imageVectorDimensions,
      async (client) => {
        for (const chunk of chunks) {
          await client.query(
            `INSERT INTO knowledge_chunks
            (id, document_id, workspace_id, chunk_index, content, modality, citation, asset_ids, embedding)
           VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::vector)`,
            [
              chunk.id,
              chunk.documentId,
              chunk.workspaceId,
              chunk.chunkIndex,
              chunk.content,
              chunk.modality ?? 'text',
              JSON.stringify(chunk.citation ?? {}),
              JSON.stringify(chunk.assetIds ?? []),
              formatPgVector(chunk.embedding),
            ],
          );
        }
      },
    );
  }

  async upsertImageChunks(chunks: IndexedKnowledgeImageChunk[]): Promise<void> {
    if (chunks.length === 0) return;
    if (!this.imageVectorDimensions) {
      throw new Error('Image vector dimensions are not configured for the knowledge index');
    }
    await withKnowledgeTransaction(
      this.vectorDimensions,
      this.imageVectorDimensions,
      async (client) => {
        for (const chunk of chunks) {
          await client.query(
            `INSERT INTO knowledge_image_chunks
            (id, document_id, workspace_id, chunk_index, asset_id, content, image_data,
             image_mime_type, citation, embedding)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::vector)`,
            [
              chunk.id,
              chunk.documentId,
              chunk.workspaceId,
              chunk.chunkIndex,
              chunk.assetId,
              chunk.content,
              chunk.imageData ?? null,
              chunk.imageMimeType ?? null,
              JSON.stringify(chunk.citation ?? {}),
              formatPgVector(chunk.embedding),
            ],
          );
        }
      },
    );
  }

  async markDocumentReady(documentId: string, chunkCount: number): Promise<void> {
    await this.ensure();
    await getKnowledgePool().query(
      `UPDATE knowledge_documents
          SET status = 'ready', chunk_count = $2, updated_at = now()
        WHERE id = $1`,
      [documentId, chunkCount],
    );
  }

  async markDocumentFailed(documentId: string, error: string): Promise<void> {
    await this.ensure();
    await getKnowledgePool().query(
      `UPDATE knowledge_documents
          SET status = 'failed', error_message = $2, updated_at = now()
        WHERE id = $1`,
      [documentId, error],
    );
  }

  async deleteDocument(documentId: string): Promise<boolean> {
    await this.ensure();
    const result = await getKnowledgePool().query(
      'DELETE FROM knowledge_documents WHERE id = $1 AND workspace_id = $2',
      [documentId, this.workspaceId],
    );
    return (result.rowCount || 0) > 0;
  }

  async countReadyDocuments(): Promise<number> {
    await this.ensure();
    const result = await getKnowledgePool().query<{ count: string }>(
      `SELECT count(*)::text AS count FROM knowledge_documents
        WHERE workspace_id = $1 AND status = 'ready'`,
      [this.workspaceId],
    );
    return Number(result.rows[0]?.count || 0);
  }

  async queryChunks(embedding: number[], config: RagRetrievalConfig): Promise<RetrievedChunkRow[]> {
    await this.ensure();
    const result = await getKnowledgePool().query<RetrievedChunkRow>(
      `SELECT c.document_id, d.file_name, c.chunk_index, c.content,
              c.modality, c.citation, c.asset_ids,
              1 - (c.embedding <=> $1::vector) AS similarity
        FROM knowledge_chunks c
         JOIN knowledge_documents d ON d.id = c.document_id
        WHERE c.workspace_id = $2 AND d.status = 'ready'
          AND 1 - (c.embedding <=> $1::vector) >= $3
        ORDER BY c.embedding <=> $1::vector
        LIMIT $4`,
      [formatPgVector(embedding), this.workspaceId, config.minSimilarity, config.topK],
    );
    return result.rows;
  }

  async queryImageChunks(
    embedding: number[],
    config: RagRetrievalConfig,
  ): Promise<RetrievedImageChunkRow[]> {
    if (!this.imageVectorDimensions) return [];
    await this.ensure();
    const result = await getKnowledgePool().query<RetrievedImageChunkRow>(
      `SELECT c.id, c.document_id, d.file_name, c.chunk_index, c.asset_id, c.content,
              c.image_data, c.image_mime_type, c.citation,
              1 - (c.embedding <=> $1::vector) AS similarity
        FROM knowledge_image_chunks c
         JOIN knowledge_documents d ON d.id = c.document_id
        WHERE c.workspace_id = $2 AND d.status = 'ready'
          AND 1 - (c.embedding <=> $1::vector) >= $3
        ORDER BY c.embedding <=> $1::vector
        LIMIT $4`,
      [formatPgVector(embedding), this.workspaceId, config.minSimilarity, config.topK],
    );
    return result.rows;
  }

  async insertSnapshot(input: {
    id: string;
    query: string;
    config: RagRetrievalConfig;
    hits: RagHit[];
    context: string;
    selectionConfirmed: boolean;
  }): Promise<void> {
    await this.ensure();
    await getKnowledgePool().query(
      `INSERT INTO generation_rag_snapshots
        (id, workspace_id, query_text, retrieval_config, selection_confirmed, retrieved_chunks, rendered_context)
       VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb, $7)`,
      [
        input.id,
        this.workspaceId,
        input.query,
        JSON.stringify(input.config),
        input.selectionConfirmed,
        JSON.stringify(input.hits),
        input.context,
      ],
    );
  }

  async getSnapshotContext(id: string): Promise<string | undefined> {
    await this.ensure();
    const result = await getKnowledgePool().query<{ rendered_context: string }>(
      `SELECT rendered_context FROM generation_rag_snapshots
        WHERE id = $1 AND workspace_id = $2 AND selection_confirmed = true LIMIT 1`,
      [id, this.workspaceId],
    );
    return result.rows[0]?.rendered_context;
  }

  async getSnapshotEvidenceRow(id: string): Promise<
    | {
        id: string;
        query_text: string;
        retrieval_config: unknown;
        selection_confirmed: boolean;
        retrieved_chunks: unknown;
      }
    | undefined
  > {
    await this.ensure();
    const result = await getKnowledgePool().query<{
      id: string;
      query_text: string;
      retrieval_config: unknown;
      selection_confirmed: boolean;
      retrieved_chunks: unknown;
    }>(
      `SELECT id, query_text, retrieval_config, selection_confirmed, retrieved_chunks
         FROM generation_rag_snapshots
        WHERE id = $1 AND workspace_id = $2 LIMIT 1`,
      [id, this.workspaceId],
    );
    return result.rows[0];
  }

  async updateSnapshotSelection(id: string, hits: RagHit[], context: string): Promise<void> {
    await this.ensure();
    await getKnowledgePool().query(
      `UPDATE generation_rag_snapshots
          SET retrieved_chunks = $3::jsonb, rendered_context = $4, selection_confirmed = true
        WHERE id = $1 AND workspace_id = $2`,
      [id, this.workspaceId, JSON.stringify(hits), context],
    );
  }
}

export function getKnowledgeIndexProvider(
  vectorDimensions: number,
  imageVectorDimensions?: number,
): KnowledgeIndexProvider {
  const providerId = process.env.KNOWLEDGE_INDEX_PROVIDER?.trim() || 'pgvector';
  if (providerId !== 'pgvector') {
    throw new Error(`Unknown knowledge index provider: ${providerId}`);
  }
  return new PgVectorKnowledgeIndexProvider(vectorDimensions, imageVectorDimensions);
}
