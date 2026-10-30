/**
 * Autopilot: today's research funnel from candidates written to the shortlist, the pre-screen
 * that decides which candidates are worth a simulation, and the distinct alphas the Critic
 * keeps. Read-only apart from two actions: retrain the pre-screen, re-score the vault.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { RefreshCwIcon, ScaleIcon } from 'lucide-react'
import { toast } from 'sonner'
import { errorMessage } from '@/api/http'
import { cn } from '@/lib/cn'
import { DASH, fmt, isNum } from '@/lib/format'
import { useNow } from '@/lib/now'
import { useRefetchOn } from '@/lib/ws'
import {
  AUTOPILOT_KEY,
  autopilot,
  type Gate,
  type PreScreen,
  type ShortlistEntry,
  type Stage,
} from '@/screens/autopilot/api'
import {
  Badge,
  Button,
  Empty,
  ErrorNotice,
  LINK,
  Metric,
  Page,
  PageHeader,
  Panel,
  Skeleton,
  type Tone,
} from '@/ui/kit'

const POLL_MS = 15_000

export function AutopilotScreen() {
  const queryClient = useQueryClient()
  const query = useQuery({
    queryKey: AUTOPILOT_KEY,
    queryFn: autopilot.get,
    refetchInterval: POLL_MS,
  })
  useRefetchOn('agents', AUTOPILOT_KEY, 3000)
  const refresh = () => queryClient.invalidateQueries({ queryKey: AUTOPILOT_KEY })

  const train = useMutation({
    mutationFn: autopilot.train,
    onSuccess: () => {
      toast.success('Pre-screen retraining')
      void refresh()
    },
    onError: (error) => toast.error(errorMessage(error)),
  })
  const rescore = useMutation({
    mutationFn: autopilot.rescore,
    onSuccess: () => {
      toast.success('Critic re-scoring the vault')
      void refresh()
    },
    onError: (error) => toast.error(errorMessage(error)),
  })
  const data = query.data

  return (
    <Page>
      <PageHeader
        title="Autopilot"
        description="Today's research funnel, the pre-screen, and the shortlist the Critic keeps."
        actions={
          <>
            <Button
              loading={train.isPending}
              disabled={data?.prescreen.status === 'training'}
              onClick={() => train.mutate()}
            >
              <RefreshCwIcon />
              Retrain Pre-screen
            </Button>
            <Button loading={rescore.isPending} onClick={() => rescore.mutate()}>
              <ScaleIcon />
              Re-score Vault
            </Button>
          </>
        }
      />

      {query.isError && !data ? (
        <ErrorNotice error={query.error} title="Could not load the Autopilot" />
      ) : !data ? (
        <div className="grid gap-4 xl:grid-cols-2">
          <Skeleton className="h-72" label="Loading the funnel" />
          <Skeleton className="h-72" label="Loading the pre-screen" />
        </div>
      ) : (
        <>
          <div className="grid gap-4 xl:grid-cols-2">
            <FunnelPanel stages={data.funnel} />
            <PreScreenPanel prescreen={data.prescreen} />
          </div>
          <ShortlistPanel entries={data.shortlist} />
        </>
      )}
    </Page>
  )
}

// ── Funnel ──────────────────────────────────────────────────────────────────────────────

function FunnelPanel({ stages }: { stages: Stage[] }) {
  const top = stages.find((s) => isNum(s.count) && s.count > 0)?.count ?? 0
  // Log scale, so a stage of 4 still shows beside one of 5,000.
  const width = (count: number) =>
    top > 0 && count > 0 ? Math.max(2, (Math.log10(count + 1) / Math.log10(top + 1)) * 100) : 0

  return (
    <Panel
      title="Today's Funnel"
      description="From candidates written to the shortlist, since the platform day began."
    >
      <ol className="flex flex-col gap-3">
        {stages.map((stage, i) => {
          const previous = stages[i - 1]?.count
          const rate =
            isNum(stage.count) && isNum(previous) && previous > 0 ? stage.count / previous : null
          return (
            <li key={stage.key} className="flex flex-col gap-1">
              <div className="flex items-baseline justify-between gap-3">
                <div className="flex min-w-0 flex-col">
                  <span className="text-body font-medium text-ink">{stage.label}</span>
                  <span className="text-body-compact text-pretty text-ink-subtle">
                    {stage.note}
                  </span>
                </div>
                <div className="flex shrink-0 items-baseline gap-2">
                  {rate !== null && (
                    <span className="num text-body-compact text-ink-subtle">
                      {fmt.pct(rate, rate < 0.01 ? 2 : 1)}
                    </span>
                  )}
                  <span className="num text-title text-ink">
                    {isNum(stage.count) ? fmt.int(stage.count) : DASH}
                  </span>
                </div>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-surface-3" aria-hidden>
                {isNum(stage.count) && (
                  <div
                    className="h-full rounded-full bg-ink-tertiary"
                    style={{ width: `${width(stage.count)}%` }}
                  />
                )}
              </div>
            </li>
          )
        })}
      </ol>
    </Panel>
  )
}

// ── Pre-screen ──────────────────────────────────────────────────────────────────────────

const PILL: Record<string, { label: string; tone: Tone; dot: string }> = {
  active: { label: 'Active', tone: 'profit', dot: 'bg-pnl-positive' },
  watching: { label: 'Watching', tone: 'neutral', dot: 'bg-status-queued' },
  training: { label: 'Training', tone: 'neutral', dot: 'bg-status-running' },
  untrained: { label: 'Untrained', tone: 'muted', dot: 'bg-status-idle' },
  failed: { label: 'Failed', tone: 'loss', dot: 'bg-pnl-negative' },
}

function pillOf(prescreen: PreScreen) {
  if (prescreen.status === 'ready') return PILL[prescreen.active ? 'active' : 'watching']
  return PILL[prescreen.status] ?? PILL['untrained']
}

function PreScreenPanel({ prescreen }: { prescreen: PreScreen }) {
  useNow(60_000) // keeps "trained 3 min ago" current
  const pill = pillOf(prescreen)
  const lift = isNum(prescreen.lift) ? `${fmt.ratio(prescreen.lift, 1)}×` : DASH

  return (
    <Panel
      title="Pre-screen"
      description={`Writes ${fmt.int(prescreen.oversample)} candidates for every simulation and sends the best, keeping ${fmt.pct(prescreen.exploreShare, 0)} at random to keep learning.`}
      actions={
        pill && (
          <Badge tone={pill.tone} className="border border-hairline-strong">
            <span className={cn('size-1.5 rounded-full', pill.dot)} aria-hidden />
            {pill.label}
          </Badge>
        )
      }
    >
      <div className="flex flex-col gap-4">
        <p className="text-body text-pretty text-ink-muted">{prescreen.reason}</p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <Metric boxed size="sm" label="AUC" value={fmt.ratio(prescreen.auc)} />
          <Metric
            boxed
            size="sm"
            label="Lift"
            value={lift}
            tone={isNum(prescreen.lift) && prescreen.lift >= 1.3 ? 'profit' : 'neutral'}
          />
          <Metric
            boxed
            size="sm"
            label="Trained On"
            value={fmt.int(prescreen.samples)}
            hint={`${fmt.int(prescreen.positives)} passed`}
          />
          <Metric
            boxed
            size="sm"
            label="Trained"
            value={prescreen.trainedAt ? fmt.ago(prescreen.trainedAt) : DASH}
          />
          <Metric boxed size="sm" label="Screened Today" value={fmt.int(prescreen.screenedToday)} />
          <Metric boxed size="sm" label="Sent Today" value={fmt.int(prescreen.keptToday)} />
        </div>
        <p className="text-body-compact text-pretty text-ink-subtle">
          Lift is how much more often the candidates it ranks in its top tenth pass BRAIN's checks
          than candidates on average, measured on recent alphas it never trained on. At 1.0× it
          picks no better than chance.
        </p>
      </div>
    </Panel>
  )
}

// ── Shortlist ───────────────────────────────────────────────────────────────────────────

const GATE_DOT = (gate: Gate) =>
  gate.ok === true ? 'bg-pnl-positive' : gate.ok === false ? 'bg-pnl-negative' : 'bg-ink-tertiary'

function Gates({ gates }: { gates: Record<string, Gate> }) {
  return (
    <span className="flex items-center gap-1">
      {Object.entries(gates).map(([name, gate]) => (
        <span
          key={name}
          title={`${name}: ${gate.note}`}
          className={cn('size-2 rounded-full', GATE_DOT(gate))}
        />
      ))}
    </span>
  )
}

function ShortlistPanel({ entries }: { entries: ShortlistEntry[] }) {
  return (
    <Panel
      title="Shortlist"
      description="One alpha per distinct family, best Critic score first. Every one is below 0.6 correlation with your submitted pool."
      bodyClassName={entries.length ? 'p-0' : ''}
    >
      {entries.length === 0 ? (
        <Empty title="Nothing on the shortlist yet">
          The shortlist fills as the Critic passes alphas and the Family Clerk finds new families.
          Re-score the vault to start.
        </Empty>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-body">
            <thead>
              <tr className="border-b border-hairline text-left text-caption uppercase tracking-wide text-ink-subtle">
                <th className="px-4 py-2 font-medium">Alpha</th>
                <th className="px-4 py-2 text-right font-medium">Score</th>
                <th className="px-4 py-2 font-medium">Market</th>
                <th className="px-4 py-2 text-right font-medium">Pool Corr</th>
                <th className="px-4 py-2 text-right font-medium">Family</th>
                <th className="px-4 py-2 font-medium">Gates</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.alphaId} className="border-b border-hairline-subtle last:border-0">
                  <td className="px-4 py-2">
                    <Link
                      to="/alpha/$alphaId"
                      params={{ alphaId: entry.alphaId }}
                      className={cn('num', LINK)}
                    >
                      {entry.alphaId}
                    </Link>
                  </td>
                  <td className="num px-4 py-2 text-right text-ink">{fmt.ratio(entry.score, 1)}</td>
                  <td className="num px-4 py-2 text-ink-muted">
                    {[entry.region, isNum(entry.delay) ? `D${entry.delay}` : null, entry.universe]
                      .filter(Boolean)
                      .join(' · ') || DASH}
                  </td>
                  <td className="num px-4 py-2 text-right text-ink-muted">
                    {fmt.ratio(entry.poolCorrelation)}
                  </td>
                  <td className="num px-4 py-2 text-right text-ink-muted">
                    {fmt.int(entry.familySize)}
                  </td>
                  <td className="px-4 py-2">
                    <Gates gates={entry.gates} />
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
