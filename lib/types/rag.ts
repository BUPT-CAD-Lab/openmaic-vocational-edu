export interface RagHit {
  documentId: string;
  documentName: string;
  chunkIndex: number;
  score: number;
  excerpt: string;
}

export interface RagSource {
  documentId: string;
  name: string;
  score: number;
  excerptCount: number;
}

export interface RagEvidence {
  id: string;
  query: string;
  hits: RagHit[];
  sources: RagSource[];
}
