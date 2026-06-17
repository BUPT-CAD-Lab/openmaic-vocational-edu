import { chunkKnowledgeText } from './chunker';
import type { DocumentArtifact, DocumentAsset, DocumentBlock } from '@/lib/document';

export type KnowledgeUnitModality = 'text' | 'image' | 'table' | 'formula' | 'layout';

export interface KnowledgeUnitCitation {
  fileName?: string;
  mimeType?: string;
  providerId?: string;
  pageNumber?: number;
  blockId?: string;
  assetId?: string;
  label?: string;
}

export interface KnowledgeUnit {
  chunkIndex: number;
  modality: KnowledgeUnitModality;
  content: string;
  embeddingText: string;
  assetIds: string[];
  citation: KnowledgeUnitCitation;
}

export interface KnowledgeImageUnit {
  chunkIndex: number;
  assetId: string;
  content: string;
  embeddingImage: {
    id: string;
    mimeType?: string;
    data?: string;
    url?: string;
    description?: string;
  };
  imageSrc?: string;
  imageMimeType?: string;
  citation: KnowledgeUnitCitation;
}

export function sanitizeKnowledgeText(value: string | undefined): string {
  return (value ?? '')
    .replace(/\u0000/g, '')
    .replace(/[\u0001-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, ' ')
    .replace(/\r\n?/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function pageLabel(pageNumber?: number): string | undefined {
  return typeof pageNumber === 'number' && pageNumber > 0 ? `page ${pageNumber}` : undefined;
}

function citationLabel(input: {
  modality: KnowledgeUnitModality;
  block?: DocumentBlock;
  asset?: DocumentAsset;
}): string {
  const page = pageLabel(input.block?.pageNumber ?? input.asset?.pageNumber);
  const parts = [input.modality, page].filter(Boolean);
  return parts.join(', ');
}

function stringifyTable(block: DocumentBlock): string {
  const rows = block.metadata?.data;
  const caption = sanitizeKnowledgeText(
    typeof block.metadata?.caption === 'string' ? block.metadata.caption : block.text,
  );
  const tableText = Array.isArray(rows)
    ? rows
        .filter((row): row is unknown[] => Array.isArray(row))
        .map((row) => row.map((cell) => sanitizeKnowledgeText(String(cell ?? ''))).join(' | '))
        .join('\n')
    : '';
  return [caption, tableText].filter(Boolean).join('\n\n');
}

function blockText(block: DocumentBlock): string {
  if (block.type === 'table') return stringifyTable(block);
  return sanitizeKnowledgeText(block.text ?? block.html);
}

export function assetProxyText(asset: DocumentAsset): string {
  const metadataText = [
    asset.description,
    typeof asset.metadata?.caption === 'string' ? asset.metadata.caption : undefined,
    typeof asset.metadata?.ocr === 'string' ? asset.metadata.ocr : undefined,
    typeof asset.metadata?.alt === 'string' ? asset.metadata.alt : undefined,
  ];
  return sanitizeKnowledgeText(metadataText.filter(Boolean).join('\n\n'));
}

function unitHeader(artifact: DocumentArtifact, label: string): string {
  return sanitizeKnowledgeText([artifact.metadata.fileName, label].filter(Boolean).join(' - '));
}

export function documentArtifactToKnowledgeUnits(artifact: DocumentArtifact): KnowledgeUnit[] {
  const units: KnowledgeUnit[] = [];
  const imageAssetsByPage = new Map<number, DocumentAsset[]>();
  const fileName = artifact.metadata.fileName;

  for (const asset of artifact.assets) {
    if (asset.type !== 'image') continue;
    if (typeof asset.pageNumber === 'number') {
      const assets = imageAssetsByPage.get(asset.pageNumber) ?? [];
      assets.push(asset);
      imageAssetsByPage.set(asset.pageNumber, assets);
    }
  }

  const addUnit = (input: Omit<KnowledgeUnit, 'chunkIndex'>) => {
    units.push({ ...input, chunkIndex: units.length });
  };

  for (const block of artifact.blocks) {
    const modality: KnowledgeUnitModality =
      block.type === 'markdown' ? 'text' : (block.type as KnowledgeUnitModality);
    if (!['text', 'table', 'formula', 'layout'].includes(modality)) continue;

    const text = blockText(block);
    if (!text) continue;

    const attachedAssets =
      typeof block.pageNumber === 'number' ? (imageAssetsByPage.get(block.pageNumber) ?? []) : [];
    const assetIds = attachedAssets.map((asset) => asset.id);
    const label = citationLabel({ modality, block });
    const header = unitHeader(artifact, label);

    if (modality === 'text') {
      for (const chunk of chunkKnowledgeText(text)) {
        const content = [header, chunk].filter(Boolean).join('\n\n');
        addUnit({
          modality,
          content,
          embeddingText: content,
          assetIds,
          citation: {
            fileName,
            mimeType: artifact.metadata.mimeType,
            providerId: artifact.metadata.providerId,
            pageNumber: block.pageNumber,
            blockId: block.id,
            label,
          },
        });
      }
      continue;
    }

    const content = [header, text].filter(Boolean).join('\n\n');
    addUnit({
      modality,
      content,
      embeddingText: content,
      assetIds,
      citation: {
        fileName,
        mimeType: artifact.metadata.mimeType,
        providerId: artifact.metadata.providerId,
        pageNumber: block.pageNumber,
        blockId: block.id,
        label,
      },
    });
  }

  for (const asset of artifact.assets) {
    if (asset.type !== 'image') continue;
    const proxyText = assetProxyText(asset);
    if (!proxyText) continue;

    const label = citationLabel({ modality: 'image', asset });
    const header = unitHeader(artifact, label);
    const content = [header, proxyText].filter(Boolean).join('\n\n');
    addUnit({
      modality: 'image',
      content,
      embeddingText: content,
      assetIds: [asset.id],
      citation: {
        fileName,
        mimeType: artifact.metadata.mimeType,
        providerId: artifact.metadata.providerId,
        pageNumber: asset.pageNumber,
        assetId: asset.id,
        label,
      },
    });
  }

  return units;
}

export function documentArtifactToKnowledgeImageUnits(
  artifact: DocumentArtifact,
  startChunkIndex = 0,
): KnowledgeImageUnit[] {
  const units: KnowledgeImageUnit[] = [];
  const fileName = artifact.metadata.fileName;

  for (const asset of artifact.assets) {
    if (asset.type !== 'image') continue;
    if (!asset.data && typeof asset.metadata?.url !== 'string') continue;

    const proxyText = assetProxyText(asset);
    const label = citationLabel({ modality: 'image', asset });
    const header = unitHeader(artifact, label);
    const content = [header, proxyText || 'Document image'].filter(Boolean).join('\n\n');
    const url = typeof asset.metadata?.url === 'string' ? asset.metadata.url : undefined;

    units.push({
      chunkIndex: startChunkIndex + units.length,
      assetId: asset.id,
      content,
      imageSrc: asset.data ?? url,
      imageMimeType: asset.mimeType,
      embeddingImage: {
        id: asset.id,
        mimeType: asset.mimeType,
        data: asset.data,
        url,
        description: proxyText,
      },
      citation: {
        fileName,
        mimeType: artifact.metadata.mimeType,
        providerId: artifact.metadata.providerId,
        pageNumber: asset.pageNumber,
        assetId: asset.id,
        label,
      },
    });
  }

  return units;
}
