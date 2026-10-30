import { NextRequest } from 'next/server';
import { resolveRequestOwner } from '@/lib/server/identity/resolve';
import {
  attachOwnerCookies,
  invalidOwnerCredentialResponse,
} from '@/lib/server/identity/with-owner';
import { apiError, apiSuccess } from '@/lib/server/api-response';
import { callLLM } from '@/lib/ai/llm';
import { resolveStageModel } from '@/lib/server/model-config/llm';
import { requestWorkspaceId } from '@/lib/server/model-config/runtime';
import { getServerPersistenceProvider } from '@/lib/persistence/server-provider';
import { createLogger } from '@/lib/logger';

const log = createLogger('GeneratePersona');

export async function POST(req: NextRequest) {
  const owner = await resolveRequestOwner(req);
  if (!owner.ok) return invalidOwnerCredentialResponse();
  return attachOwnerCookies(await generate(req, owner.principal.ownerId), owner.setCookies);
}

interface CourseStats {
  stageId: string;
  name: string;
  generationComplete: boolean;
  chatSessions: number;
  chatUserTurns: number;
  quizSessions: number;
  quizSubmitted: number;
  quizReviewed: number;
  playbackSessions: number;
  completedSessions: number;
  totalSessions: number;
}

async function fetchLearnerHistory(ownerId: string): Promise<CourseStats[]> {
  const { pool } = await getServerPersistenceProvider(process.env.DATABASE_URL!);

  // 1. Courses owned by this learner
  const coursesRes = await pool.query<{
    stage_id: string;
    name: string;
    generation_complete: boolean;
  }>(
    `SELECT sm.stage_id, ds.name, sm.generation_complete
     FROM stage_meta sm
     JOIN document_stages ds ON ds.id = sm.stage_id
     WHERE sm.owner_id = $1 AND sm.deleted_at IS NULL`,
    [ownerId],
  );

  if (coursesRes.rows.length === 0) return [];

  const stageIds = coursesRes.rows.map((r) => r.stage_id);

  // 2. All runtime sessions for this learner across those courses
  const sessionsRes = await pool.query<{
    id: string;
    stage_id: string;
    kind: string;
    status: string;
  }>(
    `SELECT id, stage_id, kind, status
     FROM runtime_sessions
     WHERE learner_key = $1 AND stage_id = ANY($2)`,
    [ownerId, stageIds],
  );

  const sessionIds = sessionsRes.rows.map((r) => r.id);

  // 3. Quiz and chat records for submitted/reviewed phase detection and chat turn counts
  type RecordRow = { session_id: string; data: unknown };
  let recordsRows: RecordRow[] = [];
  if (sessionIds.length > 0) {
    const recordsRes = await pool.query<RecordRow>(
      `SELECT session_id, data
       FROM runtime_records
       WHERE session_id = ANY($1)
       ORDER BY seq ASC`,
      [sessionIds],
    );
    recordsRows = recordsRes.rows;
  }

  // Aggregate per course
  const statsMap = new Map<string, CourseStats>();
  for (const c of coursesRes.rows) {
    statsMap.set(c.stage_id, {
      stageId: c.stage_id,
      name: c.name,
      generationComplete: c.generation_complete,
      chatSessions: 0,
      chatUserTurns: 0,
      quizSessions: 0,
      quizSubmitted: 0,
      quizReviewed: 0,
      playbackSessions: 0,
      completedSessions: 0,
      totalSessions: 0,
    });
  }

  // Count sessions per course and kind
  for (const s of sessionsRes.rows) {
    const stat = statsMap.get(s.stage_id);
    if (!stat) continue;
    stat.totalSessions++;
    if (s.status === 'completed') stat.completedSessions++;
    if (s.kind === 'chat') stat.chatSessions++;
    else if (s.kind === 'quizAttempt') stat.quizSessions++;
    else if (s.kind === 'playback') stat.playbackSessions++;
  }

  // Build session-kind lookup for record classification
  const sessionKindMap = new Map<string, string>();
  for (const s of sessionsRes.rows) sessionKindMap.set(s.id, s.kind);

  // Aggregate records
  for (const r of recordsRows) {
    const kind = sessionKindMap.get(r.session_id);
    if (!kind) continue;

    // Find the stage for this session
    const session = sessionsRes.rows.find((s) => s.id === r.session_id);
    if (!session) continue;
    const stat = statsMap.get(session.stage_id);
    if (!stat) continue;

    const data = r.data as Record<string, unknown>;
    if (kind === 'chat') {
      // Count user turns
      if (data && typeof data === 'object' && (data as { role?: unknown }).role === 'user') {
        stat.chatUserTurns++;
      }
    } else if (kind === 'quizAttempt') {
      if (data && typeof data === 'object') {
        const phase = (data as { phase?: unknown }).phase;
        if (phase === 'submitted') stat.quizSubmitted++;
        else if (phase === 'reviewed') stat.quizReviewed++;
      }
    }
  }

  return [...statsMap.values()];
}

async function generate(req: NextRequest, ownerId: string) {
  let model;
  try {
    const workspaceId = await requestWorkspaceId(req);
    const resolved = await resolveStageModel({ stage: 'chat-adapter', workspaceId });
    model = resolved.model;
  } catch (error) {
    log.error('Failed to resolve model for persona generation:', error);
    return apiError('MISSING_MODEL', 503, 'No chat model is configured');
  }

  let courses: CourseStats[];
  try {
    courses = await fetchLearnerHistory(ownerId);
  } catch (error) {
    log.error('Failed to fetch learner history:', error);
    return apiError('INTERNAL_ERROR', 500, 'Failed to fetch learning history');
  }

  const totalCourses = courses.length;
  const completedCourses = courses.filter((c) => c.generationComplete).length;
  const activeCourses = courses.filter((c) => c.totalSessions > 0).length;

  const totalQuizSessions = courses.reduce((s, c) => s + c.quizSessions, 0);
  const totalSubmitted = courses.reduce((s, c) => s + c.quizSubmitted, 0);
  const totalReviewed = courses.reduce((s, c) => s + c.quizReviewed, 0);
  const totalChatUserTurns = courses.reduce((s, c) => s + c.chatUserTurns, 0);
  const totalPlayback = courses.reduce((s, c) => s + c.playbackSessions, 0);
  const totalCompletedSessions = courses.reduce((s, c) => s + c.completedSessions, 0);
  const totalSessions = courses.reduce((s, c) => s + c.totalSessions, 0);

  // Per-course detail lines for the LLM prompt
  const courseLines = courses
    .map((c) => {
      const parts: string[] = [`课程《${c.name}》`];
      if (c.totalSessions === 0) {
        parts.push('（未开始学习）');
      } else {
        if (c.chatSessions > 0)
          parts.push(`AI对话${c.chatSessions}次（用户发言${c.chatUserTurns}轮）`);
        if (c.quizSessions > 0) {
          const detail =
            c.quizReviewed > 0
              ? `已批阅${c.quizReviewed}次`
              : c.quizSubmitted > 0
                ? `已提交${c.quizSubmitted}次`
                : `草稿${c.quizSessions}次`;
          parts.push(`测验${c.quizSessions}次（${detail}）`);
        }
        if (c.playbackSessions > 0) parts.push(`回放${c.playbackSessions}次`);
        if (c.completedSessions > 0)
          parts.push(`完成${c.completedSessions}/${c.totalSessions}个会话`);
      }
      return parts.join('，');
    })
    .join('\n');

  const historySection =
    totalCourses === 0
      ? '该学习者尚未创建任何课程。'
      : `学习者共有 ${totalCourses} 门课程（已完成内容生成 ${completedCourses} 门，有学习记录 ${activeCourses} 门）。
总会话数：${totalSessions}，已完成：${totalCompletedSessions}。
总测验次数：${totalQuizSessions}，已提交：${totalSubmitted}，已批阅：${totalReviewed}。
AI对话用户发言总轮次：${totalChatUserTurns}，课件回放次数：${totalPlayback}。

各课程明细：
${courseLines}`;

  const prompt = `你是一个智能教育画像分析师。根据下方学习者的真实历史数据，生成一份多维度学习画像报告。

【学习历史数据】
${historySection}

【输出要求】
请按以下四个维度分析，每个维度给出一个0-100的分数，并用1-2句话说明打分依据，必须引用具体的历史事件（例如"共完成X次测验"、"在课程《XXX》中..."等），不得凭空推断。

输出格式（严格按JSON，不要加任何前缀或解释）：
{
  "dimensions": [
    {
      "key": "completion",
      "label": "结果类·完成度",
      "score": <0-100整数>,
      "summary": "<1-2句，引用具体数据>"
    },
    {
      "key": "breadth_depth",
      "label": "行为类·广度与深度",
      "score": <0-100整数>,
      "summary": "<1-2句，引用具体数据>"
    },
    {
      "key": "risk",
      "label": "风险类·学习风险",
      "score": <0-100整数>,
      "summary": "<1-2句，说明风险点或无明显风险，引用具体数据>"
    },
    {
      "key": "engagement",
      "label": "行为类·学习积极性",
      "score": <0-100整数>,
      "summary": "<1-2句，引用具体数据>"
    }
  ],
  "overall": "<2-3句综合画像，自然流畅，直接描述这位学习者的特点和建议>"
}`;

  try {
    const { text } = await callLLM(
      { model, prompt, maxOutputTokens: 800 },
      'chat-adapter',
    );

    // Parse and validate the JSON response
    const raw = text.trim();
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      log.error('LLM did not return valid JSON for persona:', raw.slice(0, 200));
      return apiError('INTERNAL_ERROR', 500, 'Failed to generate persona');
    }
    const parsed = JSON.parse(jsonMatch[0]) as {
      dimensions: { key: string; label: string; score: number; summary: string }[];
      overall: string;
    };
    return apiSuccess({ persona: parsed });
  } catch (error) {
    log.error('Persona generation failed:', error);
    return apiError('INTERNAL_ERROR', 500, 'Failed to generate persona');
  }
}
