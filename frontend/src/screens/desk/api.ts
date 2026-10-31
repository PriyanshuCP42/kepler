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

export type Readiness = 'ready' | 'check' | 'hurts' | 'fading' | 'overfit' | 'blocked'

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
  readyIds?: string[]
  readyImpact?: Effect[]
}

export const DESK_KEY = ['desk'] as const

export const desk = {
  get: () => http.get<Desk>('/api/desk'),
  /** Asks BRAIN for every missing Production Correlation, as a background job. */
  checkProd: () => http.post<{ taskId: string }>('/api/desk/prod-correlation'),
  /** Writes the best unchecked candidates' neighbours as one task in Tasks, not started. */
  guard: (limit = 20, cores = 4) =>
    http.post<{ id: number; name: string; finalists: number }>('/api/desk/guard', {
      limit,
      cores,
    }),
}
