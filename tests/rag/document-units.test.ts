import { describe, expect, it } from 'vitest';
import {
  documentArtifactToKnowledgeUnits,
  sanitizeKnowledgeText,
} from '@/lib/server/knowledge/document-units';
import type { DocumentArtifact } from '@/lib/document';

describe('documentArtifactToKnowledgeUnits', () => {
  it('keeps text, table, formula, layout, and described image evidence searchable', () => {
    const artifact: DocumentArtifact = {
      metadata: {
        fileName: 'safety.pdf',
        mimeType: 'application/pdf',
        pageCount: 2,
        providerId: 'mineru',
      },
      blocks: [
        {
          id: 'text-1',
          type: 'text',
          text: '通风橱使用前需要检查风机和防护玻璃。',
          pageNumber: 1,
        },
        {
          id: 'table-1',
          type: 'table',
          pageNumber: 1,
          metadata: {
            caption: '检查项目',
            data: [
              ['项目', '要求'],
              ['风机', '正常运行'],
            ],
          },
        },
        {
          id: 'formula-1',
          type: 'formula',
          text: 'v = q / a',
          pageNumber: 2,
        },
        {
          id: 'layout-1',
          type: 'layout',
          text: '操作步骤区域',
          pageNumber: 2,
        },
      ],
      assets: [
        {
          id: 'img-1',
          type: 'image',
          mimeType: 'image/png',
          pageNumber: 1,
          description: '通风橱面板示意图',
        },
      ],
    };

    const units = documentArtifactToKnowledgeUnits(artifact);

    expect(units.map((unit) => unit.modality)).toEqual([
      'text',
      'table',
      'formula',
      'layout',
      'image',
    ]);
    expect(units[0].assetIds).toEqual(['img-1']);
    expect(units[1].content).toContain('风机 | 正常运行');
    expect(units[2].citation).toMatchObject({ blockId: 'formula-1', pageNumber: 2 });
    expect(units[4]).toMatchObject({
      modality: 'image',
      assetIds: ['img-1'],
      citation: { assetId: 'img-1', pageNumber: 1 },
    });
  });

  it('does not create vector units for images that have no text proxy', () => {
    const artifact: DocumentArtifact = {
      metadata: { fileName: 'image-only.pdf' },
      blocks: [],
      assets: [{ id: 'img-1', type: 'image', mimeType: 'image/png' }],
    };

    expect(documentArtifactToKnowledgeUnits(artifact)).toEqual([]);
  });

  it('removes NUL bytes and unsafe control characters before PostgreSQL storage', () => {
    expect(sanitizeKnowledgeText('液压\u0000泵\u0007测试')).toBe('液压泵 测试');

    const artifact: DocumentArtifact = {
      metadata: { fileName: 'nul.pdf' },
      blocks: [
        {
          id: 'text-1',
          type: 'text',
          text: '第一步\u0000检查压力表',
        },
      ],
      assets: [],
    };

    const [unit] = documentArtifactToKnowledgeUnits(artifact);
    expect(unit.content).toContain('第一步检查压力表');
    expect(unit.content).not.toContain('\u0000');
  });
});
