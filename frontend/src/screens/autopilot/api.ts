/** Autopilot: today's research funnel, the pre-screen, and the Critic's shortlist. */

import { http } from '@/api/http'

export interface Stage {
  key: string
  label: string
  /** `null` while it is not known yet. */
  count: number | null
  note: string
}

export interface PreScreen {
  status: 'untrained' | 'training' | 'ready' | 'failed'
  /** Filtering sends right now. */
  active: boolean
  /** Why it is or is not filtering, in a sentence. */
  reason: string
  trainedAt: string | null
  samples: number
  positives: number
  auc: number | null
  /** Pass rate of the model's top 10% over the average pass rate, on held-out recent alphas. */
  lift: number | null
  oversample: number
  exploreShare: number
  screenedToday: number
  keptToday: number
}

export interface Gate {
  ok: boolean | null
  value: unknown
  note: string
}

export interface ShortlistEntry {
  alphaId: string
  score: number
  region: string | null
  delay: number | null
  universe: string | null
  poolCorrelation: number | null
  familySize: number
  gates: Record<string, Gate>
  reasons: string[]
}

export interface Autopilot {
  funnel: Stage[]
  prescreen: PreScreen
  shortlist: ShortlistEntry[]
}

export interface RepairMove {
  failure: string
  move: string
  name: string
  tries: number
  wins: number
}

export interface RepairSummary {
  /** Near misses not tried yet. */
  waiting: number
  /** Repairs written but not yet judged, then how the judged ones came out. */
  pending: number
  fixed: number
  notFixed: number
  errors: number
  /** What the Repair Engineer has learned, best first. */
  moves: RepairMove[]
}

export interface RepairTaskAdded {
  id: number
  name: string
  simulations: number
}

export const AUTOPILOT_KEY = ['autopilot'] as const
export const REPAIR_KEY = ['autopilot', 'repair'] as const

export const autopilot = {
  get: () => http.get<Autopilot>('/api/autopilot'),
  /** Retrains the pre-screen in the background. 409 while it is already training. */
  train: () => http.post<{ started: boolean }>('/api/autopilot/prescreen/train'),
  /** Scores every alpha in the vault again. 409 while the Critic is already busy. */
  rescore: () => http.post<{ started: boolean }>('/api/agents/critic/rescore'),
  repair: () => http.get<RepairSummary>('/api/agents/repair'),
  /** Writes fixes for the best near misses as one task in Tasks, not started. */
  planRepairs: (alphas = 100, perAlpha = 3, cores = 4) =>
    http.post<RepairTaskAdded>('/api/agents/repair/tasks', { alphas, perAlpha, cores }),
}
