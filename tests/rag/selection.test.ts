import { describe, expect, it } from 'vitest';
import {
  filterSelectedRagHits,
  mergeRetrievedKnowledgeRows,
} from '@/lib/server/knowledge/repository';
import type { RagHit } from '@/lib/types/rag';
import type { RetrievedChunkRow, RetrievedImageChunkRow } from '@/lib/server/knowledge/postgres';

const candidates: RagHit[] = [
  {
    id: 'manual-a:text:1',
    documentId: 'manual-a',
    documentName: 'manual.pdf',
    chunkIndex: 1,
    score: 0.91,
    excerpt: '步骤 A',
  },
  {
    id: 'manual-a:image:1',
    documentId: 'manual-a',
    documentName: 'manual.pdf',
    chunkIndex: 1,
    score: 0.83,
    excerpt: '步骤 B 图片',
    modality: 'image',
  },
  {
    documentId: 'manual-b',
    documentName: 'service.pdf',
    chunkIndex: 0,
    score: 0.74,
    excerpt: '警告 C',
  },
];

describe('RAG excerpt selection', () => {
  it('keeps only excerpts explicitly selected by the user', () => {
    expect(
      filterSelectedRagHits(candidates, [
        { id: 'manual-a:image:1', documentId: 'manual-a', chunkIndex: 1 },
        { documentId: 'manual-b', chunkIndex: 0 },
      ]),
    ).toEqual([candidates[1], candidates[2]]);
  });

  it('does not allow unknown client keys to inject new excerpts', () => {
    expect(filterSelectedRagHits(candidates, [{ documentId: 'manual-x', chunkIndex: 99 }])).toEqual(
      [],
    );
  });
});

describe('RAG multimodal retrieval merge', () => {
  it('reserves image evidence slots when image retrieval returns candidates', () => {
    const textRows: RetrievedChunkRow[] = Array.from({ length: 6 }, (_, index) => ({
      document_id: 'doc-a',
      file_name: 'manual.pdf',
      chunk_index: index,
      content: `text ${index}`,
      similarity: 0.99 - index * 0.01,
      modality: 'text',
      citation: null,
      asset_ids: null,
    }));
    const imageRows: RetrievedImageChunkRow[] = [
      {
        id: 'image-a',
        document_id: 'doc-a',
        file_name: 'manual.pdf',
        chunk_index: 6,
        asset_id: 'asset-a',
        content: 'image a',
        image_data: 'data:image/png;base64,aaa',
        image_mime_type: 'image/png',
        similarity: 0.5,
        citation: null,
      },
      {
        id: 'image-b',
        document_id: 'doc-a',
        file_name: 'manual.pdf',
        chunk_index: 7,
        asset_id: 'asset-b',
        content: 'image b',
        image_data: 'data:image/png;base64,bbb',
        image_mime_type: 'image/png',
        similarity: 0.49,
        citation: null,
      },
    ];

    const merged = mergeRetrievedKnowledgeRows(textRows, imageRows, 6);

    expect(merged).toHaveLength(6);
    expect(merged.filter((row) => row.kind === 'image')).toHaveLength(2);
    expect(merged.slice(0, 2).map((row) => row.kind)).toEqual(['image', 'image']);
  });
});
