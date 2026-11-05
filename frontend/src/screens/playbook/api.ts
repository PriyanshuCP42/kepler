/** The Playbook: WorldQuant's guidance and this account's lessons, kept on this machine. */

import { http } from '@/api/http'

export interface Insight {
  id: string
  title: string
  summary: string
  source: string
  date: string
  topics: string[]
  /** Kepler screens the insight belongs on. */
  places: string[]
  points: string[]
  /** [label, value] pairs. */
  figures: string[][]
  actions: string[]
  /** BM25 relevance to the question; absent when the list was not searched. */
  score?: number | null
}

export interface PlaybookView {
  insights: Insight[]
  topics: string[]
  places: string[]
  total: number
  path: string
}

export interface PlaybookQuery {
  q?: string
  topic?: string
  place?: string
  limit?: number
}

export const PLAYBOOK_KEY = ['playbook'] as const

export const playbookApi = {
  search: (query: PlaybookQuery = {}) => {
    const params = new URLSearchParams()
    if (query.q) params.set('q', query.q)
    if (query.topic) params.set('topic', query.topic)
    if (query.place) params.set('place', query.place)
    if (query.limit) params.set('limit', String(query.limit))
    const qs = params.toString()
    return http.get<PlaybookView>(`/api/playbook${qs ? `?${qs}` : ''}`)
  },
}
