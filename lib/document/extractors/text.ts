import type { DocumentExtractorInput, DocumentExtractorProvider } from '../types';

const TEXT_MIME_TYPES = ['text/plain', 'text/markdown', 'text/html'];

function stripHtml(html: string): string {
  return html
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/(p|div|section|article|h[1-6]|li|tr)>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function decodeText(input: DocumentExtractorInput): string {
  const text = input.buffer.toString('utf8');
  return input.mimeType.toLowerCase() === 'text/html' ? stripHtml(text) : text.trim();
}

export const textDocumentExtractorProvider: DocumentExtractorProvider = {
  id: 'plain-text',
  displayName: 'Plain Text',
  supportedMimeTypes: TEXT_MIME_TYPES,
  capabilities: {
    text: true,
    images: false,
    tables: false,
    formulas: false,
    layout: false,
    ocr: false,
    async: false,
  },
  async extract(input) {
    return {
      metadata: {
        fileName: input.fileName,
        fileSize: input.fileSize,
        mimeType: input.mimeType,
        providerId: 'plain-text',
      },
      blocks: [
        {
          id: 'document-text',
          type: input.mimeType.toLowerCase() === 'text/markdown' ? 'markdown' : 'text',
          text: decodeText(input),
        },
      ],
      assets: [],
      diagnostics: [],
    };
  },
};
