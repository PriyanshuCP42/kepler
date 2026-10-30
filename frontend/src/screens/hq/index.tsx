/**
 * Research HQ: every agent of the research pipeline at its desk in a three-floor building,
 * carrying work to the next one as it happens. Live from `/api/agents` and its two topics;
 * demo data until the backend serves them.
 */

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { ApiError, http } from '@/api/http'
import { cn } from '@/lib/cn'
import { fmt } from '@/lib/format'
import { useTopic } from '@/lib/ws'
import { Badge, Empty, ErrorNotice, KV, Notice, Page, PageHeader, Panel } from '@/ui/kit'
import { demoHandoff, demoSnapshot } from './demo'
import { HqScene } from './scene'
import { type Agent, type AgentsSnapshot, FLOOR_NAMES, type Handoff } from './types'

const KEY = ['agents'] as const
const FEED = 20
/** Handoffs kept in the snapshot, so a picked agent can list its own. */
const KEPT = 200
const DEMO_EVERY_MS = 1400

function withHandoff(snapshot: AgentsSnapshot, handoff: Handoff): AgentsSnapshot {
  if (snapshot.handoffs.some((h) => h.id === handoff.id)) return snapshot
  return { ...snapshot, handoffs: [...snapshot.handoffs, handoff].slice(-KEPT) }
}

export function HqScreen() {
  const queryClient = useQueryClient()
  const host = useRef<HTMLDivElement>(null)
  const scene = useRef<HqScene | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [demo, setDemo] = useState<AgentsSnapshot | null>(null)

  const agents = useQuery({
    queryKey: KEY,
    queryFn: () => http.get<AgentsSnapshot>('/api/agents'),
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 2,
  })
  const missing = agents.error instanceof ApiError && agents.error.status === 404
  const snapshot = missing ? demo : (agents.data ?? null)

  useTopic('agents', (payload) => queryClient.setQueryData(KEY, payload as AgentsSnapshot))
  useTopic('agent-handoff', (payload) => {
    const handoff = payload as Handoff
    scene.current?.handoff(handoff)
    queryClient.setQueryData<AgentsSnapshot>(KEY, (old) => (old ? withHandoff(old, handoff) : old))
  })

  // Demo mode: the twelve agents, and a made-up handoff every second and a half.
  useEffect(() => {
    if (!missing) return
    setDemo(demoSnapshot())
    const timer = setInterval(() => {
      const handoff = demoHandoff()
      scene.current?.handoff(handoff)
      setDemo((old) => {
        if (!old) return old
        const next = withHandoff(old, handoff)
        return {
          ...next,
          agents: next.agents.map((a) =>
            a.id === handoff.from || a.id === handoff.to
              ? {
                  ...a,
                  state: a.status === 'live' ? 'working' : 'idle',
                  lastEventAt: handoff.at,
                  stats:
                    a.id === handoff.from
                      ? { ...a.stats, handedOn: Number(a.stats['handedOn'] ?? 0) + handoff.count }
                      : a.stats,
                }
              : a,
          ),
        }
      })
    }, DEMO_EVERY_MS)
    return () => clearInterval(timer)
  }, [missing])

  useEffect(() => {
    const element = host.current
    if (!element) return
    const built = new HqScene(element, {
      onPick: setSelected,
      reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    })
    scene.current = built
    return () => {
      built.dispose()
      scene.current = null
    }
  }, [])

  const roster = snapshot?.agents
  useEffect(() => {
    if (roster) scene.current?.setAgents(roster)
  }, [roster])
  useEffect(() => {
    scene.current?.select(selected)
  }, [selected])

  const byId = useMemo(
    () => new Map((snapshot?.agents ?? []).map((a) => [a.id, a] as const)),
    [snapshot?.agents],
  )
  const picked = selected ? byId.get(selected) : undefined
  const feed = (snapshot?.handoffs ?? []).slice(-FEED).reverse()
  const live = snapshot?.agents.filter((a) => a.status === 'live').length ?? 0

  return (
    <Page>
      <PageHeader
        title="Research HQ"
        description="Every agent in the research pipeline at its desk. Each box is real work being handed to the next agent. Drag to turn the building; click an agent for its details."
        actions={
          snapshot && (
            <>
              {missing && <Badge tone="warn">Demo data</Badge>}
              <Badge tone="outline">
                {live} of {snapshot.agents.length} agents live
              </Badge>
            </>
          )
        }
      />
      {missing && (
        <Notice tone="warn" title="Showing demo data">
          The backend does not serve the agents yet, so these handoffs are made up. The building
          switches to live data as soon as it does.
        </Notice>
      )}
      {agents.isError && !missing && (
        <ErrorNotice error={agents.error} title="Could not load the agents" />
      )}

      <div className="grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div
          ref={host}
          className="relative h-[clamp(420px,62vh,720px)] min-w-0 overflow-hidden rounded-lg border border-hairline bg-canvas"
          role="img"
          aria-label="A three-floor building with the research agents at their desks, passing data boxes to each other."
        />
        <AgentPanel agent={picked} handoffs={snapshot?.handoffs ?? []} names={byId} />
      </div>

      <div className="grid min-w-0 gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <Panel title="Live Feed" description="The last handoffs, newest first.">
          {feed.length === 0 ? (
            <Empty title="No handoffs yet">They appear here as the agents pass work on.</Empty>
          ) : (
            <ul className="flex flex-col divide-y divide-hairline">
              {feed.map((h) => (
                <li key={h.id} className="flex items-center gap-3 py-1.5 text-body">
                  <span className="num w-16 shrink-0 text-ink-subtle">
                    {new Date(h.at).toLocaleTimeString([], { hour12: false })}
                  </span>
                  <button
                    type="button"
                    className="min-w-0 truncate text-left text-ink hover:text-link"
                    onClick={() => setSelected(h.from)}
                  >
                    {byId.get(h.from)?.name ?? h.from}
                  </button>
                  <span className="text-ink-tertiary">→</span>
                  <button
                    type="button"
                    className="min-w-0 truncate text-left text-ink hover:text-link"
                    onClick={() => setSelected(h.to)}
                  >
                    {byId.get(h.to)?.name ?? h.to}
                  </button>
                  <span className="ml-auto shrink-0 truncate text-ink-subtle">{h.label}</span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
        <Panel title="Legend">
          <ul className="flex flex-col gap-2 text-body text-ink-muted">
            <LegendItem dot="bg-pnl-positive" text="Working" />
            <LegendItem dot="bg-ink-tertiary" text="Idle" />
            <LegendItem dot="bg-pnl-negative" text="Blocked" />
            <LegendItem
              dot="bg-ink-tertiary opacity-40"
              text="Joining soon: planned, not built yet"
            />
            <li className="flex items-start gap-2">
              <span className="mt-1 size-2.5 shrink-0 rounded-xs bg-primary" />
              <span>
                A data box. Agents on one floor carry it by hand; between floors it rides the lift
                on the right.
              </span>
            </li>
          </ul>
          <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-body-compact">
            {([3, 2, 1] as const).map((floor) => (
              <div key={floor} className="contents">
                <dt className="num text-ink-subtle">Floor {floor}</dt>
                <dd className="text-ink">{FLOOR_NAMES[floor]}</dd>
              </div>
            ))}
          </dl>
        </Panel>
      </div>
    </Page>
  )
}

function LegendItem({ dot, text }: { dot: string; text: string }) {
  return (
    <li className="flex items-center gap-2">
      <span className={cn('size-2 shrink-0 rounded-full', dot)} />
      {text}
    </li>
  )
}

function AgentPanel({
  agent,
  handoffs,
  names,
}: {
  agent: Agent | undefined
  handoffs: Handoff[]
  names: Map<string, Agent>
}) {
  if (!agent) {
    return (
      <Panel title="Agent">
        <Empty title="Click an agent">
          Its role, what it is doing and the work it handed on show here.
        </Empty>
      </Panel>
    )
  }
  const own = handoffs
    .filter((h) => h.from === agent.id || h.to === agent.id)
    .slice(-8)
    .reverse()
  const stats = Object.entries(agent.stats)
  return (
    <Panel
      title={agent.name}
      description={agent.role}
      actions={
        agent.status === 'planned' ? (
          <Badge tone="outline">Joining soon</Badge>
        ) : (
          <Badge
            tone={
              agent.state === 'working' ? 'profit' : agent.state === 'blocked' ? 'loss' : 'muted'
            }
          >
            {agent.state === 'working' ? 'Working' : agent.state === 'blocked' ? 'Blocked' : 'Idle'}
          </Badge>
        )
      }
    >
      <div className="flex flex-col gap-4">
        {agent.detail && <p className="text-body text-ink-muted">{agent.detail}</p>}
        <KV
          items={[
            ['Floor', `${agent.floor} · ${FLOOR_NAMES[agent.floor]}`],
            ['Last active', agent.lastEventAt ? fmt.ago(agent.lastEventAt) : '—'],
            ...stats.map(
              ([key, value]) =>
                [label(key), typeof value === 'number' ? fmt.int(value) : value] as [
                  string,
                  string,
                ],
            ),
          ]}
        />
        <div className="flex flex-col gap-1.5">
          <h3 className="text-caption font-medium text-ink-subtle uppercase">Recent handoffs</h3>
          {own.length === 0 ? (
            <p className="text-body-compact text-ink-subtle">None yet.</p>
          ) : (
            <ul className="flex flex-col gap-1 text-body-compact">
              {own.map((h) => (
                <li key={h.id} className="flex gap-2">
                  <span className={h.from === agent.id ? 'text-ink-subtle' : 'text-link'}>
                    {h.from === agent.id ? 'to' : 'from'}
                  </span>
                  <span className="min-w-0 truncate text-ink">
                    {names.get(h.from === agent.id ? h.to : h.from)?.name ?? h.to}
                  </span>
                  <span className="ml-auto shrink-0 text-ink-subtle">{h.label}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </Panel>
  )
}

/** `handedOn` → "Handed on". */
function label(key: string): string {
  const words = key.replace(/([A-Z])/g, ' $1').toLowerCase()
  return words.charAt(0).toUpperCase() + words.slice(1)
}
