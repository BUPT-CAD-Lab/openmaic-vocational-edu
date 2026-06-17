import { createHash, randomUUID } from 'node:crypto';
import {
  documentArtifactToKnowledgeImageUnits,
  documentArtifactToKnowledgeUnits,
  sanitizeKnowledgeText,
  type KnowledgeUnitCitation,
} from './document-units';
import { getEmbeddingProvider, getImageEmbeddingProvider } from './embedding';
import {
  getKnowledgeIndexProvider,
  type DocumentRow,
  type RetrievedChunkRow,
  type RetrievedImageChunkRow,
} from './postgres';
import type { RagEvidence, RagHit, RagRetrievalConfig, RagSource } from '@/lib/types/rag';
import { normalizeRagRetrievalConfig } from '@/lib/rag/config';
import type { DocumentArtifact } from '@/lib/document';

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

function getKnowledgeProviders() {
  const embedding = getEmbeddingProvider();
  const imageEmbedding = getImageEmbeddingProvider();
  const index = getKnowledgeIndexProvider(embedding.dimensions, imageEmbedding?.dimensions);
  return { embedding, imageEmbedding, index };
}

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
  const { index } = getKnowledgeProviders();
  return (await index.listDocuments()).map(toDocument);
}

export async function findDocumentByHash(hash: string): Promise<KnowledgeDocument | undefined> {
  const { index } = getKnowledgeProviders();
  const row = await index.findDocumentByHash(hash);
  return row ? toDocument(row) : undefined;
}

export function hashDocument(buffer: Buffer): string {
  return createHash('sha256').update(buffer).digest('hex');
}

export async function createKnowledgeDocument(input: {
  fileName: string;
  mimeType: string;
  buffer: Buffer;
  artifact: DocumentArtifact;
  pageCount?: number;
}): Promise<KnowledgeDocument> {
  const hash = hashDocument(input.buffer);
  const existing = await findDocumentByHash(hash);
  if (existing) return existing;

  const { embedding, imageEmbedding, index } = getKnowledgeProviders();
  const id = randomUUID();
  await index.insertDocument({
    id,
    workspaceId: index.workspaceId,
    fileName: input.fileName,
    mimeType: input.mimeType,
    fileSize: input.buffer.length,
    contentHash: hash,
    data: input.buffer,
    parsedText: sanitizeKnowledgeText(
      input.artifact.blocks
        .map((block) => block.text)
        .filter(Boolean)
        .join('\n\n'),
    ),
    pageCount: input.pageCount,
    embeddingModel: imageEmbedding
      ? `${embedding.id}:${embedding.model}; image=${imageEmbedding.id}:${imageEmbedding.model}`
      : `${embedding.id}:${embedding.model}`,
  });

  try {
    const units = documentArtifactToKnowledgeUnits(input.artifact);
    if (units.length === 0)
      throw new Error('No searchable content was extracted from this document');
    const embeddings: number[][] = [];
    const embeddingTexts = units.map((unit) => unit.embeddingText);
    for (let start = 0; start < embeddingTexts.length; start += EMBEDDING_BATCH_SIZE) {
      embeddings.push(
        ...(await embedding.embed(embeddingTexts.slice(start, start + EMBEDDING_BATCH_SIZE))),
      );
    }
    await index.upsertChunks(
      units.map((unit) => ({
        id: randomUUID(),
        documentId: id,
        workspaceId: index.workspaceId,
        chunkIndex: unit.chunkIndex,
        content: sanitizeKnowledgeText(unit.content),
        modality: unit.modality,
        citation: unit.citation,
        assetIds: unit.assetIds,
        embedding: embeddings[unit.chunkIndex],
      })),
    );
    let imageUnitCount = 0;
    if (imageEmbedding) {
      const imageUnits = documentArtifactToKnowledgeImageUnits(input.artifact, units.length);
      imageUnitCount = imageUnits.length;
      if (imageUnits.length > 0) {
        const imageEmbeddings: number[][] = [];
        for (let start = 0; start < imageUnits.length; start += EMBEDDING_BATCH_SIZE) {
          imageEmbeddings.push(
            ...(await imageEmbedding.embedImages(
              imageUnits
                .slice(start, start + EMBEDDING_BATCH_SIZE)
                .map((unit) => unit.embeddingImage),
            )),
          );
        }
        await index.upsertImageChunks(
          imageUnits.map((unit, unitIndex) => ({
            id: randomUUID(),
            documentId: id,
            workspaceId: index.workspaceId,
            chunkIndex: unit.chunkIndex,
            assetId: unit.assetId,
            content: sanitizeKnowledgeText(unit.content),
            imageData: unit.imageSrc,
            imageMimeType: unit.imageMimeType,
            citation: unit.citation,
            embedding: imageEmbeddings[unitIndex],
          })),
        );
      }
    }
    await index.markDocumentReady(id, units.length + imageUnitCount);
  } catch (error) {
    await index.markDocumentFailed(id, error instanceof Error ? error.message : String(error));
    throw error;
  }

  const documents = await listKnowledgeDocuments();
  const document = documents.find((record) => record.id === id);
  if (!document) throw new Error('Stored knowledge document could not be read back');
  return document;
}

export async function deleteKnowledgeDocument(id: string): Promise<boolean> {
  const { index } = getKnowledgeProviders();
  return index.deleteDocument(id);
}

export interface RagSnapshot {
  id: string;
  context: string;
  config: RagRetrievalConfig;
  hits: RagHit[];
  sources: RagSource[];
}

type RetrievedKnowledgeRow =
  | {
      kind: 'text';
      row: RetrievedChunkRow;
      score: number;
    }
  | {
      kind: 'image';
      row: RetrievedImageChunkRow;
      score: number;
    };
type RetrievedTextKnowledgeRow = Extract<RetrievedKnowledgeRow, { kind: 'text' }>;
type RetrievedImageKnowledgeRow = Extract<RetrievedKnowledgeRow, { kind: 'image' }>;

export function mergeRetrievedKnowledgeRows(
  textRows: RetrievedChunkRow[],
  imageRows: RetrievedImageChunkRow[],
  topK: number,
): RetrievedKnowledgeRow[] {
  const normalizedTopK = Math.max(0, Math.floor(topK));
  if (normalizedTopK === 0) return [];

  const textCandidates: RetrievedTextKnowledgeRow[] = textRows
    .map((row) => ({ kind: 'text' as const, row, score: Number(row.similarity) }))
    .sort((left, right) => right.score - left.score);
  const imageCandidates: RetrievedImageKnowledgeRow[] = imageRows
    .map((row) => ({ kind: 'image' as const, row, score: Number(row.similarity) }))
    .sort((left, right) => right.score - left.score);

  if (imageCandidates.length === 0 || normalizedTopK === 1) {
    return [...textCandidates, ...imageCandidates]
      .sort((left, right) => right.score - left.score)
      .slice(0, normalizedTopK);
  }

  const imageSlots = Math.min(imageCandidates.length, Math.max(1, Math.ceil(normalizedTopK / 3)));
  const selectedImages = imageCandidates.slice(0, imageSlots);
  const selectedImageIds = new Set(selectedImages.map((candidate) => candidate.row.id));
  const remaining = [
    ...textCandidates,
    ...imageCandidates.filter((candidate) => !selectedImageIds.has(candidate.row.id)),
  ]
    .sort((left, right) => right.score - left.score)
    .slice(0, normalizedTopK - selectedImages.length);

  return [...selectedImages, ...remaining];
}

function buildSnapshotMaterial(hits: RagHit[]): { context: string; sources: RagSource[] } {
  const sources = new Map<string, RagSource>();
  const excerpts = hits.map((hit) => {
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
    const parts = [
      `Source: ${hit.documentName}`,
      `excerpt ${hit.chunkIndex + 1}`,
      hit.modality ? `modality: ${hit.modality}` : undefined,
      hit.citation?.pageNumber ? `page: ${hit.citation.pageNumber}` : undefined,
      hit.citation?.label ? `location: ${hit.citation.label}` : undefined,
    ].filter(Boolean);
    return sanitizeKnowledgeText(`[${parts.join(', ')}]\n${hit.excerpt}`);
  });
  return {
    context: sanitizeKnowledgeText(
      `## Retrieved Knowledge Base Material\nUse these configured knowledge excerpts as factual grounding. Do not invent technical specifications or procedures that conflict with them.\n\n${excerpts.join('\n\n')}`,
    ),
    sources: Array.from(sources.values()),
  };
}

export async function retrieveKnowledgeSnapshot(
  query: string,
  requestedConfig?: Partial<RagRetrievalConfig>,
): Promise<RagSnapshot | undefined> {
  const config = normalizeRagRetrievalConfig(requestedConfig);
  const { embedding, imageEmbedding, index } = getKnowledgeProviders();
  if ((await index.countReadyDocuments()) === 0) {
    throw new Error(
      'No indexed knowledge materials are ready. Upload course material in the knowledge base first.',
    );
  }

  const [queryEmbedding] = await embedding.embed([query]);
  const textRows = await index.queryChunks(queryEmbedding, config);
  const imageRows =
    imageEmbedding && index.imageVectorDimensions
      ? await imageEmbedding
          .embedText([query])
          .then(([imageQueryEmbedding]) => index.queryImageChunks(imageQueryEmbedding, config))
      : [];
  if (textRows.length === 0 && imageRows.length === 0) return undefined;

  const rows = mergeRetrievedKnowledgeRows(textRows, imageRows, config.topK);

  let usedChars = 0;
  const hits: RagHit[] = [];
  for (const result of rows) {
    const row = result.row;
    const header = `[Source: ${row.file_name}, excerpt ${row.chunk_index + 1}]`;
    const remaining = config.maxContextChars - usedChars - header.length - 2;
    if (remaining <= 0) break;
    const excerpt = sanitizeKnowledgeText(row.content).slice(0, remaining);
    usedChars += header.length + excerpt.length + 2;
    if (result.kind === 'image') {
      const imageRow = result.row;
      hits.push({
        id: imageRow.id,
        documentId: imageRow.document_id,
        documentName: imageRow.file_name,
        chunkIndex: imageRow.chunk_index,
        score: Number(imageRow.similarity),
        excerpt,
        modality: 'image',
        citation: parseCitation(imageRow.citation),
        assetIds: [imageRow.asset_id],
        imageSrc: imageRow.image_data ?? undefined,
        imageMimeType: imageRow.image_mime_type ?? undefined,
      });
    } else {
      const textRow = result.row;
      hits.push({
        id: `${textRow.document_id}:text:${textRow.chunk_index}`,
        documentId: textRow.document_id,
        documentName: textRow.file_name,
        chunkIndex: textRow.chunk_index,
        score: Number(textRow.similarity),
        excerpt,
        modality: textRow.modality ?? undefined,
        citation: parseCitation(textRow.citation),
        assetIds: parseAssetIds(textRow.asset_ids),
      });
    }
  }
  const { context, sources } = buildSnapshotMaterial(hits);
  const id = randomUUID();
  await index.insertSnapshot({
    id,
    query,
    config,
    hits,
    context,
    selectionConfirmed: false,
  });
  return { id, context, config, hits, sources };
}

export async function getRagSnapshotContext(id: string): Promise<string | undefined> {
  const { index } = getKnowledgeProviders();
  return index.getSnapshotContext(id);
}

export async function getRagSnapshotEvidence(id: string): Promise<RagEvidence | undefined> {
  const { index } = getKnowledgeProviders();
  const row = await index.getSnapshotEvidenceRow(id);
  if (!row) return undefined;

  const rawHits = Array.isArray(row.retrieved_chunks) ? row.retrieved_chunks : [];
  const hits = rawHits.filter(isRagHit);
  const { sources } = buildSnapshotMaterial(hits);

  return {
    id: row.id,
    query: row.query_text,
    config: normalizeRagRetrievalConfig(row.retrieval_config),
    selectionConfirmed: row.selection_confirmed,
    hits,
    sources,
  };
}

export function filterSelectedRagHits(
  candidates: RagHit[],
  selectedHits: Array<Pick<RagHit, 'documentId' | 'chunkIndex'> & { id?: string }>,
): RagHit[] {
  const selectedKeys = new Set(
    selectedHits.map((hit) => hit.id ?? `${hit.documentId}:${hit.chunkIndex}`),
  );
  return candidates.filter((hit) =>
    selectedKeys.has(hit.id ?? `${hit.documentId}:${hit.chunkIndex}`),
  );
}

export async function selectRagSnapshotHits(
  id: string,
  selectedHits: Array<Pick<RagHit, 'documentId' | 'chunkIndex'> & { id?: string }>,
): Promise<RagEvidence | undefined> {
  const evidence = await getRagSnapshotEvidence(id);
  if (!evidence) return undefined;

  const hits = filterSelectedRagHits(evidence.hits, selectedHits);
  if (hits.length === 0) {
    throw new Error('Select at least one retrieved excerpt before continuing');
  }

  const { context, sources } = buildSnapshotMaterial(hits);
  const { index } = getKnowledgeProviders();
  await index.updateSnapshotSelection(id, hits, context);

  return { ...evidence, selectionConfirmed: true, hits, sources };
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

function parseCitation(value: unknown): KnowledgeUnitCitation | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as KnowledgeUnitCitation)
    : undefined;
}

function parseAssetIds(value: unknown): string[] | undefined {
  return Array.isArray(value) && value.every((item) => typeof item === 'string')
    ? value
    : undefined;
}
