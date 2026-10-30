/** Demo data for before the backend serves `/api/agents`: the twelve agents and made-up handoffs. */

import type { Agent, AgentsSnapshot, Handoff } from './types'

type Seed = [
  id: string,
  name: string,
  role: string,
  floor: Agent['floor'],
  slot: number,
  live: boolean,
]

const ROSTER: Seed[] = [
  ['conductor', 'Conductor', 'Plans the day, paces the quota, restarts stalled work', 3, 0, true],
  ['strategist', 'Strategist', 'Chooses which niches get today’s simulations', 3, 1, false],
  ['hypothesis', 'Hypothesis Engine', 'Writes an economic idea, then a template', 3, 2, false],
  ['writers', 'Candidate Writers', 'The labs: turn templates into candidate alphas', 2, 0, true],
  ['prescreen', 'Pre-screen', 'Ranks candidates; only the best are simulated', 2, 1, false],
  ['validator', 'Validator', 'Checks every expression offline before it is sent', 2, 2, true],
  ['brain', 'BRAIN Gateway', 'Sends simulations to WorldQuant BRAIN and tracks them', 2, 3, true],
  ['critic', 'Critic', 'Gates and scores every finished alpha', 1, 0, true],
  ['families', 'Family Clustering', 'Groups passing alphas by PnL correlation', 1, 1, true],
  ['repair', 'Repair Agent', 'Fixes near misses from the metric that failed', 1, 2, false],
  ['selector', 'Selector', 'Picks the day’s submission set', 1, 3, false],
  ['learning', 'Learning Layer', 'Turns outcomes into rewards and retrains', 1, 4, false],
]

/** The pipeline's edges, with a typical batch size for each. */
const EDGES: [from: string, to: string, kind: string, count: number][] = [
  ['conductor', 'strategist', 'plan', 1],
  ['strategist', 'hypothesis', 'niches', 6],
  ['hypothesis', 'writers', 'templates', 12],
  ['writers', 'prescreen', 'candidates', 400],
  ['prescreen', 'validator', 'shortlist', 40],
  ['validator', 'brain', 'requests', 40],
  ['brain', 'critic', 'alphas', 30],
  ['critic', 'families', 'passed', 2],
  ['critic', 'repair', 'near misses', 3],
  ['repair', 'validator', 'fixes', 3],
  ['families', 'selector', 'elites', 1],
  ['selector', 'learning', 'outcomes', 1],
  ['learning', 'strategist', 'rewards', 1],
]

export function demoSnapshot(): AgentsSnapshot {
  return {
    agents: ROSTER.map(([id, name, role, floor, slot, live]) => ({
      id,
      name,
      role,
      floor,
      slot,
      status: live ? 'live' : 'planned',
      state: 'idle',
      detail: live ? 'Demo data' : 'Not built yet',
      stats: { handedOn: 0 },
      lastEventAt: null,
    })),
    handoffs: [],
  }
}

let counter = 0

/** One plausible handoff, with some noise in the size. */
export function demoHandoff(): Handoff {
  const [from, to, kind, size] = EDGES[Math.floor(Math.random() * EDGES.length)] ?? [
    'validator',
    'brain',
    'requests',
    40,
  ]
  const count = Math.max(1, Math.round(size * (0.5 + Math.random())))
  counter += 1
  return {
    id: `demo-${counter}`,
    from,
    to,
    kind,
    count,
    label: `${count} ${kind}`,
    at: new Date().toISOString(),
  }
}
