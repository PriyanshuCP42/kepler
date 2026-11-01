/** The Genius tracker: your Genius figures, typed in, against the Submission Desk's estimate. */

import { http } from '@/api/http'

export interface Reading {
  id: number
  readOn: string
  combined: number | null
  powerPool: number | null
  signals: number | null
  pyramids: number | null
  note: string | null
  /** `brain` when synced from BRAIN's Genius page, `you` when typed in. */
  source?: 'brain' | 'you'
  level?: string | null
  rank?: number | null
  /** What the Desk's estimate said that day, from the alphas submitted by then. */
  estimateCombined: number | null
  estimatePowerPool: number | null
  alphas: number
}

export interface Step {
  since: string
  until: string
  submitted: string[]
  brainCombined: number | null
  brainPowerPool: number | null
  estimateCombined: number | null
  estimatePowerPool: number | null
  /** How much the estimate's combined-performance score moved, as a share. */
  estimateQuality?: number | null
}

export interface Bar {
  metric: string
  label: string
  /** Expert, Master, Grandmaster. */
  bars: number[]
  value: number | null
  level: string | null
}

export interface GeniusView {
  readings: Reading[]
  steps: Step[]
  agreement: {
    pairs: number
    sameDirection: number
    correlation: number | null
    verdict: string
  }
  bars: Bar[]
  levels: string[]
  today: { combined: number | null; powerPool: number | null; alphas: number } | null
  minPairs: number
  /** Absent from a backend older than this screen. */
  syncedAt?: string | null
  syncError?: string | null
  syncHours?: number
}

export interface NewReading {
  readOn: string
  combined: number | null
  powerPool: number | null
  signals: number | null
  pyramids: number | null
  note: string | null
}

export interface OsmosisPick {
  alphaId: string
  points: number
  current: number
  recentSharpe: number | null
  turnover: number | null
  /** `chosen`, `filler` (makes up the ten) or `left_out`. */
  role: 'chosen' | 'filler' | 'left_out'
  reason: string
}

export interface OsmosisScope {
  region: string
  delay: number
  alphas: number
  eligible: boolean
  shortOf: number
  currentlyValid: boolean
  currentPoints: number
  currentAlphas: number
  currentSharpe: number | null
  proposedSharpe: number | null
  picks: OsmosisPick[]
}

export interface OsmosisView {
  locksAt: string
  minScopes: number
  minAlphas: number
  points: number
  validNow: number
  eligible: number
  scopes: OsmosisScope[]
  advice: string[]
}

export interface OsmosisApplied {
  changed: number
  failed: string[]
  view: OsmosisView
}

export interface LevelRow {
  key: string
  label: string
  you: number | null
  bar: number | null
  peers: number | null
  lowerBetter: boolean
  met: boolean | null
  note: string
}

export interface LevelPlan {
  level: string
  quarterEnd: string
  daysLeft: number
  performanceCutoff: string
  bars: LevelRow[]
  tieBreakers: LevelRow[]
  performances: Record<string, number | null>
  peersCounted: number
  pace: string[]
}

export const GENIUS_KEY = ['genius'] as const
export const LEVEL_KEY = ['genius', 'level'] as const
export const OSMOSIS_KEY = ['genius', 'osmosis'] as const

export const geniusApi = {
  get: () => http.get<GeniusView>('/api/genius'),
  add: (body: NewReading) => http.post<GeniusView>('/api/genius', body),
  /** Reads your Genius figures from BRAIN now, as today's reading. */
  sync: () => http.post<GeniusView>('/api/genius/sync'),
  remove: (id: number) => http.del<GeniusView>(`/api/genius/${id}`),
  /** Your figures this quarter against a level's bars and the consultants already at it. */
  level: (level = 'Master') => http.get<LevelPlan>(`/api/genius/level?level=${level}`),
  /** Your Osmosis points as BRAIN holds them, and a valid allocation to propose. Reads only. */
  osmosis: () => http.get<OsmosisView>('/api/genius/osmosis'),
  /** Writes the proposal for these scopes (`REGION/DELAY`) to BRAIN. */
  applyOsmosis: (scopes: string[]) =>
    http.post<OsmosisApplied>('/api/genius/osmosis/apply', { scopes }),
}
