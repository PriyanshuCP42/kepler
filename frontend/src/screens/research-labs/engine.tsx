/**
 * Template Engine: what the Dataset Doctor sees in each region, the recipe it would write for
 * each dataset, and how every recipe has done. The Daily Autopilot and the Strategist write
 * engine tasks on their own; the button here writes them now.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { useState } from 'react'
import type { components } from '@/api/generated'
import { http } from '@/api/http'
import { fmt } from '@/lib/format'
import {
  Badge,
  Button,
  Empty,
  ErrorNotice,
  LINK,
  Notice,
  Page,
  PageHeader,
  Panel,
  Segmented,
  Skeleton,
} from '@/ui/kit'

type Diagnosis = components['schemas']['Diagnosis']
type Scoreboard = components['schemas']['Scoreboard']
type Planned = components['schemas']['Planned']

const REGIONS = ['USA', 'GLB', 'ASI', 'EUR'] as const
type Region = (typeof REGIONS)[number]

const SHAPE: Record<string, string> = {
  count: 'Count',
  level: 'Level',
  estimate: 'Estimate',
  surprise: 'Surprise',
  probability: 'Probability',
}

const TH = 'py-1.5 pr-3 font-medium'
const TD = 'py-1.5 pr-3'

function ScoreboardPanel() {
  const board = useQuery({
    queryKey: ['engine', 'scoreboard'],
    queryFn: () => http.get<Scoreboard>('/api/engine/scoreboard'),
    refetchInterval: 5 * 60 * 1000,
  })
  const rows = board.data?.records ?? []
  return (
    <Panel
      title="Scoreboard"
      description={
        board.data
          ? `A trial passes when it clears every local test of the Submit Gate. A recipe with no pass in ${board.data.retireAfter} trials in a region is retired there.`
          : undefined
      }
    >
      {board.isPending ? (
        <Skeleton className="h-24" label="Loading the scoreboard" />
      ) : board.isError ? (
        <ErrorNotice error={board.error} />
      ) : !rows.length ? (
        <Empty title="No engine trials yet">
          Results appear here as engine tasks finish their simulations.
        </Empty>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-body-compact">
            <thead className="text-left text-ink-subtle">
              <tr className="border-b border-hairline">
                <th className={TH}>Recipe</th>
                <th className={TH}>Shape</th>
                <th className={TH}>Region</th>
                <th className={`${TH} text-right`}>Trials</th>
                <th className={`${TH} text-right`}>Passed</th>
                <th className={`${TH} text-right`}>Pass Rate</th>
                <th className="py-1.5 font-medium">State</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={`${r.recipe}-${r.shape}-${r.region}`}
                  className="border-b border-hairline last:border-0"
                >
                  <td className={TD} title={r.recipe}>
                    {r.title}
                  </td>
                  <td className={`${TD} text-ink-muted`}>{SHAPE[r.shape] ?? r.shape}</td>
                  <td className={`${TD} text-ink-muted`}>{r.region}</td>
                  <td className={`num ${TD} text-right`}>{fmt.int(r.tries)}</td>
                  <td className={`num ${TD} text-right`}>{fmt.int(r.passes)}</td>
                  <td className={`num ${TD} text-right`}>{fmt.pct(r.rate)}</td>
                  <td className="py-1.5">
                    {r.retired ? (
                      <Badge tone="loss">Retired</Badge>
                    ) : (
                      <Badge tone="outline">Active</Badge>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}

function DoctorPanel() {
  const [region, setRegion] = useState<Region>('USA')
  const queryClient = useQueryClient()
  const doctor = useQuery({
    queryKey: ['engine', 'doctor', region],
    queryFn: () => http.get<Diagnosis[]>(`/api/engine/doctor?region=${region}`),
    staleTime: 10 * 60 * 1000,
  })
  const plan = useMutation({
    mutationFn: () => http.post<Planned>('/api/engine/plan', { region, count: 3 }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['lab-tasks'] }),
  })
  const rows = doctor.data ?? []
  return (
    <Panel
      title="Dataset Doctor"
      description="Each promising dataset's usable, uncrowded fields, grouped by what kind of signal they are, most valuable first. Writing tasks skips datasets your submissions in the region already use."
      actions={
        <>
          <Segmented
            label="Region"
            items={REGIONS.map((r) => ({ value: r, label: r }))}
            value={region}
            onChange={(r) => {
              setRegion(r)
              plan.reset()
            }}
          />
          <Button variant="primary" loading={plan.isPending} onClick={() => plan.mutate()}>
            Write 3 Tasks
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-3">
        {plan.isError && <ErrorNotice error={plan.error} />}
        {plan.data && (
          <Notice tone={plan.data.added.length ? 'info' : 'warn'}>
            {plan.data.added.length ? (
              <>
                Wrote {plan.data.added.length} {region} tasks, not started. The Daily Autopilot
                starts them, or start them from{' '}
                <Link to="/tasks" className={LINK}>
                  Tasks
                </Link>
                .
              </>
            ) : (
              `No new ${region} dataset to try: each is submitted, busy, or retired.`
            )}
          </Notice>
        )}
        {doctor.isPending ? (
          <Skeleton className="h-40" label="Reading the catalog" />
        ) : doctor.isError ? (
          <ErrorNotice error={doctor.error} />
        ) : !rows.length ? (
          <Empty title={`Nothing usable in ${region}`}>
            Sync the {region} catalog from Data first.
          </Empty>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-body-compact">
              <thead className="text-left text-ink-subtle">
                <tr className="border-b border-hairline">
                  <th className={TH}>Dataset</th>
                  <th className={TH}>Category</th>
                  <th className={TH}>Shape</th>
                  <th className={`${TH} text-right`}>Fields</th>
                  <th className={`${TH} text-right`}>Priority</th>
                  <th className={TH}>Notes</th>
                  <th className="py-1.5 font-medium">Recipes</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((d) => (
                  <tr
                    key={`${d.datasetId}-${d.shape}`}
                    className="border-b border-hairline last:border-0"
                  >
                    <td className={TD}>{d.datasetId}</td>
                    <td className={`${TD} text-ink-muted`}>{d.categoryId}</td>
                    <td className={TD}>
                      <Badge tone="outline">{SHAPE[d.shape] ?? d.shape}</Badge>
                    </td>
                    <td className={`num ${TD} text-right`}>{fmt.int(d.fields)}</td>
                    <td className={`num ${TD} text-right`}>{fmt.ratio(d.priority, 2)}</td>
                    <td className={`${TD} text-ink-subtle`}>{d.notes.join(' · ')}</td>
                    <td className="py-1.5 text-ink-muted">{d.recipes.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Panel>
  )
}

export function EngineScreen() {
  return (
    <Page>
      <PageHeader
        title="Template Engine"
        description="Generic templates passed 0.1–5% of the time here. The engine reads what kind of signal each field is, writes the recipe fitted to that shape and region, and learns which recipes reach the Submit Gate."
      />
      <ScoreboardPanel />
      <DoctorPanel />
    </Page>
  )
}
