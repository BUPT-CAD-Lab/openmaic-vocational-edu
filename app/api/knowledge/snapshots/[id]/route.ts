import { apiError, apiSuccess } from '@/lib/server/api-response';
import { createLogger } from '@/lib/logger';
import { getRagSnapshotEvidence } from '@/lib/server/knowledge/repository';

const log = createLogger('Knowledge Snapshot API');

export const runtime = 'nodejs';

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    const evidence = await getRagSnapshotEvidence(id);
    if (!evidence) return apiError('INVALID_REQUEST', 404, 'Knowledge snapshot not found');
    return apiSuccess({ evidence });
  } catch (error) {
    log.error('Failed to read knowledge snapshot:', error);
    return apiError(
      'INTERNAL_ERROR',
      500,
      error instanceof Error ? error.message : 'Failed to read knowledge snapshot',
    );
  }
}
