/** The Submission Desk: what each shortlisted alpha would do to your pool if submitted. */

import { http } from '@/api/http'

export interface Figures {
  sharpe: number | null
  fitness: number | null
  returns: number
  drawdown: number
  turnover: number
  margin: number
}

/** How much better the pool's combined performance gets, as a share (0.02 is 2% better):
 *  Sharpe, Fitness, margin, turnover and drawdown together, better always positive. */
export interface Impact {
  total: number
  parts: Partial<Record<'sharpe' | 'fitness' | 'margin' | 'turnover' | 'drawdown', number>>
  window: string
}

export interface Effect {
  /** `all` for your whole pool, else the partition, e.g. `USA · D1 · Regular`. */
  scope: string
  members: number
  /** `null` for a partition none of your alphas is in yet. */
  before: Figures | null
  after: Figures | null
  /** The same over the recent years, the nearest to what BRAIN's Combined Alpha Performance
   *  is measured on. */
  recentBefore: Figures | null
  recentAfter: Figures | null
  /** Absent from a backend older than this screen. */
  impact?: Impact | null
}

export interface YearSharpe {
  year: number
  sharpe: number | null
}

export interface Pyramid {
  name: string
  submitted: number
  completes: boolean
  multiplier: number | null
}

export type Readiness =
  | 'ready'
  | 'check'
  | 'hurts'
  | 'fading'
  | 'overfit'
  | 'off_theme'
  | 'fails_check'
  | 'blocked'

export interface CheckLine {
  name: string
  result: string
  value: number | null
  limit: number | null
}

/** BRAIN's own submission check, as last kept. */
export interface SubmissionCheck {
  status: 'not_run' | 'passes' | 'fails'
  checkedAt: string | null
  failing: CheckLine[]
  /** Thresholds missed but only WARNING. */
  warned: CheckLine[]
  /** BRAIN asks for a Power Pool description it does not have yet. */
  needsDescription: boolean
}

export interface NeighbourResult {
  kind: string
  name: string
  state: 'queued' | 'done' | 'failed'
  sharpe: number | null
  testSharpe: number | null
  /** Kept enough of the finalist's Sharpe; `null` when it cannot be told. */
  holds: boolean | null
}

export interface GuardView {
  status: 'untested' | 'running' | 'robust' | 'fragile' | 'inconclusive'
  held: number
  measured: number
  neighbours: NeighbourResult[]
  /** Simulations of the same idea, numbers aside, in its region. */
  siblings: number
  /** The test Sharpe the best of that many variants reaches by chance. */
  luckBar: number
  years: number
  positiveYears: number
  testDays: number | null
  testWithoutSpike: number | null
  recentYears: YearSharpe[]
  recentSharpe: number | null
  fullSharpe: number | null
}

/** What to do about a candidate BRAIN's submission check refused, decided by rule. */
export interface Fix {
  action: 'describe' | 'repair' | 'drop' | 'repaired'
  reason: string
  checks: string[]
}

/** The alpha this one repairs, and the fix that made it. */
export interface Lineage {
  origin: string
  move: string
  name: string
}

/** One of this month's Power Pool themes. */
export interface PowerPoolTheme {
  id: string
  name: string
  multiplier: number
  region: string | null
  delay: number | null
  qualifier: string | null
}

/** How BRAIN can take an alpha: regular, or only as pure Power Pool in a theme market. */
export interface PowerPoolRoute {
  missed: string[]
  theme: string | null
}

export interface Candidate {
  alphaId: string
  region: string | null
  delay: number | null
  universe: string | null
  expression: string | null
  powerPool: boolean
  criticScore: number
  sharpe: number | null
  fitness: number | null
  trainSharpe: number | null
  testSharpe: number | null
  poolCorrelation: number | null
  /** BRAIN's, against every user's alphas; `null` until asked. */
  prodCorrelation: number | null
  effects: Effect[]
  guard: GuardView
  /** Absent from a backend older than this screen. */
  submission?: SubmissionCheck | null
  fix?: Fix | null
  repairOf?: Lineage | null
  route?: PowerPoolRoute | null
  pyramid: Pyramid | null
  readiness: Readiness
  reasons: string[]
}

export interface Desk {
  candidates: Candidate[]
  submitted: number
  unmeasured: number
  prodCeiling: number
  selfCeiling: number
  /** Your whole pool's combined Sharpe in each backtest year. */
  poolYears: YearSharpe[]
  recentFrom: number
  recentTo: number
  /** Submitting every Ready candidate together: your whole pool, then your Power Pool alphas,
   *  before and after. Absent from a backend older than this screen. */
  themes?: PowerPoolTheme[]
  readyIds?: string[]
  readyImpact?: Effect[]
}

export const DESK_KEY = ['desk'] as const

export const desk = {
  get: () => http.get<Desk>('/api/desk'),
  /** Asks BRAIN for every missing Production Correlation, as a background job. */
  checkProd: () => http.post<{ taskId: string }>('/api/desk/prod-correlation'),
  /** Runs BRAIN's submission check on every Ready or Check First candidate not checked yet. */
  submissionCheck: () => http.post<{ taskId: string }>('/api/desk/submission-check'),
  /** One repair round for this candidate alone, when the rules call it fixable. */
  repairOne: (alphaId: string, cores = 4) =>
    http.post<{ id: number; name: string; alphas: number }>(
      `/api/desk/repair/${encodeURIComponent(alphaId)}`,
      { cores },
    ),
  /** Writes repairs for the candidates BRAIN's submission check refused. */
  repairFailing: (cores = 4) =>
    http.post<{ id: number; name: string; alphas: number }>('/api/desk/repair', { cores }),
  /** Writes the best unchecked candidates' neighbours as one task in Tasks, not started. */
  guard: (limit = 20, cores = 4) =>
    http.post<{ id: number; name: string; finalists: number }>('/api/desk/guard', {
      limit,
      cores,
    }),
}
