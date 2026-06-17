import { NextRequest } from 'next/server';
import { extractDocument, selectDocumentExtractorProvider } from '@/lib/document';
import { apiError, apiSuccess } from '@/lib/server/api-response';
import { assertKnowledgeBackendReady } from '@/lib/server/knowledge/config';
import { createKnowledgeDocument, listKnowledgeDocuments } from '@/lib/server/knowledge/repository';
import {
  isServerConfiguredProvider,
  resolvePDFApiKey,
  resolvePDFBaseUrl,
} from '@/lib/server/provider-config';
import { createLogger } from '@/lib/logger';

const log = createLogger('Knowledge Documents API');
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const KNOWLEDGE_DOCUMENT_EXTRACTOR_PROVIDER =
  process.env.KNOWLEDGE_DOCUMENT_EXTRACTOR_PROVIDER?.trim() || undefined;

const MIME_BY_EXTENSION: Record<string, string> = {
  pdf: 'application/pdf',
  txt: 'text/plain',
  md: 'text/markdown',
  markdown: 'text/markdown',
  html: 'text/html',
  htm: 'text/html',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
};

export const runtime = 'nodejs';
export const maxDuration = 300;

export async function GET() {
  try {
    assertKnowledgeBackendReady();
    const documents = await listKnowledgeDocuments();
    return apiSuccess({ documents });
  } catch (error) {
    log.error('Failed to list knowledge documents:', error);
    return apiError(
      'INTERNAL_ERROR',
      500,
      error instanceof Error ? error.message : 'Failed to list knowledge documents',
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    assertKnowledgeBackendReady();
    const formData = await req.formData();
    const material = formData.get('file') ?? formData.get('pdf');
    if (!(material instanceof File)) {
      return apiError('MISSING_REQUIRED_FIELD', 400, 'Course material file is required');
    }
    if (material.size > MAX_UPLOAD_BYTES) {
      return apiError('INVALID_REQUEST', 400, 'Course material file must not exceed 50 MB');
    }

    const mimeType = inferMimeType(material);
    let provider;
    try {
      provider = selectDocumentExtractorProvider({
        mimeType,
        preferredProviderId:
          mimeType === 'application/pdf' ? KNOWLEDGE_DOCUMENT_EXTRACTOR_PROVIDER : undefined,
        requiredCapabilities: { text: true },
      });
    } catch (error) {
      return apiError(
        'INVALID_REQUEST',
        400,
        error instanceof Error
          ? error.message
          : `No document extractor supports MIME type "${mimeType}"`,
      );
    }

    const buffer = Buffer.from(await material.arrayBuffer());
    const extractorConfig = buildExtractorConfig(provider.id);
    if (provider.id === 'mineru' && !extractorConfig.baseUrl) {
      return apiError(
        'INVALID_REQUEST',
        422,
        `${material.name} requires a configured self-hosted MinerU extractor. Set a MinerU base URL in server provider config or PDF_MINERU_BASE_URL before uploading DOCX/PPTX materials.`,
      );
    }
    const artifact = await extractDocument({
      buffer,
      fileName: material.name,
      fileSize: material.size,
      mimeType,
      config: extractorConfig,
    });
    const document = await createKnowledgeDocument({
      fileName: material.name,
      mimeType,
      buffer,
      artifact,
      pageCount: artifact.metadata.pageCount,
    });
    return apiSuccess({ document }, 201);
  } catch (error) {
    log.error('Failed to ingest knowledge document:', error);
    return apiError(
      'INTERNAL_ERROR',
      500,
      error instanceof Error ? error.message : 'Failed to ingest knowledge document',
    );
  }
}

function inferMimeType(file: File): string {
  if (file.type) return file.type.toLowerCase();
  const extension = file.name.split('.').pop()?.toLowerCase();
  return extension
    ? (MIME_BY_EXTENSION[extension] ?? 'application/octet-stream')
    : 'application/octet-stream';
}

function buildExtractorConfig(providerId: string) {
  if (providerId !== 'unpdf' && providerId !== 'mineru' && providerId !== 'mineru-cloud') {
    return { providerId };
  }

  const managed = isServerConfiguredProvider('pdf', providerId);
  return {
    providerId,
    apiKey: resolvePDFApiKey(providerId, undefined),
    baseUrl: resolvePDFBaseUrl(providerId, managed ? undefined : undefined),
  };
}
