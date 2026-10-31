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
  /** What it predicts: a Critic pass, or BRAIN's checks when Critic passes are too thin. */
  target: 'critic' | 'brain'
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

export interface LabYield {
  lab: string
  /** What its tasks maximised, e.g. Critic Score or Train Sharpe. */
  objective: string
  tasks: number
  /** Finished simulations the Critic judged, over the last 14 days. */
  simulations: number
  passed: number
  nearMisses: number
  perThousand: number
}

export interface Autopilot {
  funnel: Stage[]
  prescreen: PreScreen
  yields: LabYield[]
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

export interface Niche {
  region: string
  delay: number
  category: string
  categoryName: string
  multiplier: number | null
  /** Alphas submitted in it this quarter; three complete the pyramid. */
  submitted: number
  synced: boolean
  simulations: number
  passed: number
  families: number
  recentSimulations: number
  recentPassed: number
  status: 'ready' | 'untried' | 'stalled' | 'not_downloaded'
  value: number
  /** New families per 100 simulations, learned. */
  expected: number
  score: number
  share: number
  boost: number
  reasons: string[]
  simulationsPlanned: number | null
}

export interface StrategistView {
  plan: Niche[]
  niches: Niche[]
}

export interface StrategistTasks {
  added: { id: number; niche: string; simulations: number; datasets: string[]; universe: string }[]
  skipped: { niche: string; reason: string }[]
}

export const AUTOPILOT_KEY = ['autopilot'] as const
export const STRATEGIST_KEY = ['autopilot', 'strategist'] as const
export const REPAIR_KEY = ['autopilot', 'repair'] as const

export const autopilot = {
  get: () => http.get<Autopilot>('/api/autopilot'),
  /** Retrains the pre-screen in the background. 409 while it is already training. */
  train: () => http.post<{ started: boolean }>('/api/autopilot/prescreen/train'),
  /** Scores every alpha in the vault again. 409 while the Critic is already busy. */
  rescore: () => http.post<{ started: boolean }>('/api/agents/critic/rescore'),
  repair: () => http.get<RepairSummary>('/api/agents/repair'),
  strategist: () => http.get<StrategistView>('/api/agents/strategist'),
  /** One Search Lab task per planned niche, not started. */
  writePlan: (niches = 5, simulations = 1500, cores = 4) =>
    http.post<StrategistTasks>('/api/agents/strategist/tasks', { niches, simulations, cores }),
  /** Writes fixes for the best near misses as one task in Tasks, not started. */
  planRepairs: (alphas = 100, perAlpha = 3, cores = 4) =>
    http.post<RepairTaskAdded>('/api/agents/repair/tasks', { alphas, perAlpha, cores }),
}
