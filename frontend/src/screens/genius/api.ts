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

export const GENIUS_KEY = ['genius'] as const

export const geniusApi = {
  get: () => http.get<GeniusView>('/api/genius'),
  add: (body: NewReading) => http.post<GeniusView>('/api/genius', body),
  /** Reads your Genius figures from BRAIN now, as today's reading. */
  sync: () => http.post<GeniusView>('/api/genius/sync'),
  remove: (id: number) => http.del<GeniusView>(`/api/genius/${id}`),
}
