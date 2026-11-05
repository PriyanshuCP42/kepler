/**
 * Road to a Genius level: each bar you must clear this quarter, how far you are from it, and
 * under it the Playbook insights that move it. The knowledge, organised by what it is for.
 */

import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { fmt } from '@/lib/format'
import { geniusApi, type LevelRow } from '@/screens/genius/api'
import { Badge, Disclosure, ErrorNotice, Notice, Panel, Segmented, Skeleton } from '@/ui/kit'
import { type Insight, PLAYBOOK_KEY, playbookApi } from './api'
import { InsightBody } from './insight'

const LEVELS = ['Expert', 'Master', 'Grandmaster'] as const
type Level = (typeof LEVELS)[number]

/** A bar's lever: the Playbook insights tagged with it are what move it. */
const LEVER_OF: Record<string, string> = {
  signals: 'signals',
  pyramids: 'pyramids',
  performance: 'performance',
}

const LEVER_TITLE: Record<string, string> = {
  tiebreakers: 'Tie-breakers: what decides among those who clear the bars',
  competitions: 'Competitions: AQP, Power Pool themes',
  quality: 'Alpha quality: the craft under every bar',
}

const show = (row: LevelRow, v: number | null) =>
  row.key === 'performance' ? fmt.ratio(v) : fmt.int(v)

function Lever({ insights }: { insights: Insight[] }) {
  if (!insights.length) {
    return <p className="text-body-compact text-ink-subtle">No insight for this yet.</p>
  }
  return (
    <div className="flex flex-col gap-1.5">
      {insights.map((i) => (
        <Disclosure key={i.id} summary={i.title}>
          <InsightBody insight={i} />
        </Disclosure>
      ))}
    </div>
  )
}

export function RoadPanel() {
  const [level, setLevel] = useState<Level>('Grandmaster')
  const plan = useQuery({
    queryKey: ['genius', 'level', level],
    queryFn: () => geniusApi.level(level),
    retry: false,
  })
  const book = useQuery({
    queryKey: [...PLAYBOOK_KEY, 'all'],
    queryFn: () => playbookApi.search({ limit: 200 }),
  })
  const by = (lever: string) => (book.data?.insights ?? []).filter((i) => i.levers?.includes(lever))
  const data = plan.data

  return (
    <Panel
      title={`Road to ${level}`}
      description={
        data
          ? `${data.daysLeft} days to ${fmt.date(data.quarterEnd)}; combined performance counts what is in by ${fmt.date(data.performanceCutoff)}. Each bar with the insights that move it.`
          : 'Each Genius bar this quarter, how far you are from it, and the insights that move it.'
      }
      actions={
        <Segmented
          label="Level"
          items={LEVELS.map((l) => ({ value: l, label: l }))}
          value={level}
          onChange={setLevel}
        />
      }
      bodyClassName="flex flex-col gap-4"
    >
      {plan.isError ? (
        <ErrorNotice error={plan.error} title="Could not read your Genius figures" />
      ) : !data ? (
        <Skeleton className="h-48" label="Reading BRAIN's leaderboard" />
      ) : (
        <>
          {data.bars.map((row) => {
            const gap = row.you != null && row.bar != null ? Math.max(0, row.bar - row.you) : null
            return (
              <section
                key={row.key}
                className="flex flex-col gap-2 rounded-md border border-hairline p-3"
              >
                <header className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-title font-medium text-ink">{row.label}</h3>
                  <div className="flex flex-wrap items-center gap-2 text-body-compact">
                    <span className="text-ink-subtle">
                      You <span className="num text-ink">{show(row, row.you)}</span> · bar{' '}
                      <span className="num text-ink">{show(row, row.bar)}</span> · {level} median{' '}
                      <span className="num text-ink">{show(row, row.peers)}</span>
                    </span>
                    {row.met ? (
                      <Badge tone="profit">Cleared</Badge>
                    ) : (
                      <Badge tone="warn">
                        {gap != null ? `${show(row, gap)} to go` : 'Not yet'}
                      </Badge>
                    )}
                  </div>
                </header>
                <Lever insights={by(LEVER_OF[row.key] ?? row.key)} />
              </section>
            )
          })}
          {data.pace.map((p) => (
            <Notice key={p} tone="info" title={p} />
          ))}
          {Object.entries(LEVER_TITLE).map(([lever, title]) => (
            <section
              key={lever}
              className="flex flex-col gap-2 rounded-md border border-hairline p-3"
            >
              <h3 className="text-title font-medium text-ink">{title}</h3>
              <Lever insights={by(lever)} />
            </section>
          ))}
        </>
      )}
    </Panel>
  )
}
