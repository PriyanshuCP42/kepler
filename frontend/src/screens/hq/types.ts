/** The Research HQ contract, as `GET /api/agents` and the `agents` / `agent-handoff` topics send it. */

export type Floor = 1 | 2 | 3

export interface Agent {
  id: string
  name: string
  role: string
  floor: Floor
  /** 0-based position on its floor, left to right. */
  slot: number
  status: 'live' | 'planned'
  state: 'idle' | 'working' | 'blocked'
  detail: string
  stats: Record<string, number | string>
  lastEventAt: string | null
}

export interface Handoff {
  id: string
  from: string
  to: string
  kind: string
  count: number
  label: string
  at: string
}

/** Handoffs newest last. */
export interface AgentsSnapshot {
  agents: Agent[]
  handoffs: Handoff[]
}

export const FLOOR_NAMES: Record<Floor, string> = {
  3: 'Planning',
  2: 'Production',
  1: 'Judgement',
}
