/**
 * Autopilot: today's research funnel from candidates written to the shortlist, the pre-screen
 * that decides which candidates are worth a simulation, and the distinct alphas the Critic
 * keeps. Read-only apart from two actions: retrain the pre-screen, re-score the vault.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate } from '@tanstack/react-router'
import { CompassIcon, RefreshCwIcon, ScaleIcon, WrenchIcon } from 'lucide-react'
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
  type Niche,
  type PreScreen,
  REPAIR_KEY,
  type RepairSummary,
  type ShortlistEntry,
  STRATEGIST_KEY,
  type Stage,
} from '@/screens/autopilot/api'
import {
  Badge,
  Button,
  Disclosure,
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
          <StrategistPanel />
          <RepairPanel />
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
            // The shortlist counts all time, so a rate against today's families means nothing.
            stage.key !== 'shortlist' && isNum(stage.count) && isNum(previous) && previous > 0
              ? stage.count / previous
              : null
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

// ── Strategist ──────────────────────────────────────────────────────────────────────────

const NICHE_STATUS: Record<Niche['status'], { label: string; tone: Tone }> = {
  ready: { label: 'Ready', tone: 'neutral' },
  untried: { label: 'Untried', tone: 'warn' },
  stalled: { label: 'Stalled', tone: 'loss' },
  not_downloaded: { label: 'No data', tone: 'muted' },
}

function StrategistPanel() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const query = useQuery({
    queryKey: STRATEGIST_KEY,
    queryFn: autopilot.strategist,
    refetchInterval: 5 * 60_000,
  })
  const write = useMutation({
    mutationFn: () => autopilot.writePlan(),
    onSuccess: (result) => {
      void queryClient.invalidateQueries({ queryKey: ['lab-tasks'] })
      const sims = result.added.reduce((sum, t) => sum + t.simulations, 0)
      const skipped = result.skipped.map((s) => `${s.niche}: ${s.reason}`).join(' ')
      toast.success(
        `${fmt.int(result.added.length)} Search Lab tasks added, ${fmt.int(sims)} simulations`,
        {
          description: `Not started; run them from Tasks.${skipped ? ` Skipped ${skipped}` : ''}`,
          action: { label: 'Open Tasks', onClick: () => void navigate({ to: '/tasks' }) },
        },
      )
    },
    onError: (error) => toast.error(errorMessage(error)),
  })
  const data = query.data

  return (
    <Panel
      title="Strategist"
      description="Which pyramids get the next simulations: worth to your Genius bars × families each one has produced, with stalled niches cooled and one place kept for an untried one."
      actions={
        <Button
          variant="primary"
          loading={write.isPending}
          disabled={!data || data.plan.length === 0}
          onClick={() => write.mutate()}
        >
          <CompassIcon />
          Write Plan Tasks
        </Button>
      }
    >
      {query.isError && !data ? (
        <ErrorNotice error={query.error} title="Could not load the Strategist" />
      ) : !data ? (
        <Skeleton className="h-40" label="Building the niche map" />
      ) : (
        <div className="flex flex-col gap-4">
          <NicheTable niches={data.plan} planned />
          <Disclosure summary={`Niche map · ${fmt.int(data.niches.length)} pyramids`}>
            <NicheTable niches={data.niches.slice(0, 40)} />
          </Disclosure>
        </div>
      )}
    </Panel>
  )
}

function NicheTable({ niches, planned = false }: { niches: Niche[]; planned?: boolean }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-body-compact">
        <thead>
          <tr className="text-left text-caption text-ink-subtle">
            <th className="py-1.5 pr-3 font-medium">Pyramid</th>
            <th className="py-1.5 pr-3 text-right font-medium">Multiplier</th>
            <th className="py-1.5 pr-3 text-right font-medium">This Quarter</th>
            <th className="py-1.5 pr-3 text-right font-medium">Simulated</th>
            <th className="py-1.5 pr-3 text-right font-medium">Families</th>
            <th className="py-1.5 pr-3 text-right font-medium">Per 100</th>
            <th className="py-1.5 pr-3 font-medium">Status</th>
            <th className="py-1.5 text-right font-medium">{planned ? 'Planned' : 'Score'}</th>
          </tr>
        </thead>
        <tbody>
          {niches.map((n) => {
            const status = NICHE_STATUS[n.status]
            return (
              <tr
                key={`${n.region}:${n.delay}:${n.category}`}
                className="border-t border-hairline"
                title={n.reasons.join(' ')}
              >
                <td className="py-1.5 pr-3 text-ink">
                  {n.region} · D{n.delay} · {n.categoryName}
                </td>
                <td className="py-1.5 pr-3 text-right num">
                  {isNum(n.multiplier) ? `${fmt.ratio(n.multiplier, 1)}×` : DASH}
                </td>
                <td className="py-1.5 pr-3 text-right num">{fmt.int(n.submitted)} / 3</td>
                <td className="py-1.5 pr-3 text-right num">{fmt.int(n.simulations)}</td>
                <td className="py-1.5 pr-3 text-right num">{fmt.int(n.families)}</td>
                <td className="py-1.5 pr-3 text-right num">{fmt.ratio(n.expected, 2)}</td>
                <td className="py-1.5 pr-3">
                  <Badge tone={status.tone}>{status.label}</Badge>
                </td>
                <td className="py-1.5 text-right num">
                  {planned
                    ? `${fmt.int(n.simulationsPlanned ?? 0)} · ${fmt.pct(n.share, 0)}`
                    : fmt.ratio(n.score, 2)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── Repairs ─────────────────────────────────────────────────────────────────────────────

function RepairPanel() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const query = useQuery({
    queryKey: REPAIR_KEY,
    queryFn: autopilot.repair,
    refetchInterval: POLL_MS,
  })
  const plan = useMutation({
    mutationFn: () => autopilot.planRepairs(),
    onSuccess: (added) => {
      void queryClient.invalidateQueries({ queryKey: REPAIR_KEY })
      void queryClient.invalidateQueries({ queryKey: ['lab-tasks'] })
      toast.success(`Repair task added: ${fmt.int(added.simulations)} simulations`, {
        description: 'It is not started. Run it from Tasks.',
        action: { label: 'Open Tasks', onClick: () => void navigate({ to: '/tasks' }) },
      })
    },
    onError: (error) => toast.error(errorMessage(error)),
  })
  const data = query.data

  return (
    <Panel
      title="Repairs"
      description="Near misses re-run with the fix most likely to clear what failed. The Repair Engineer learns which fix works for which failure."
      actions={
        <Button
          variant="primary"
          loading={plan.isPending}
          disabled={!data || data.waiting === 0}
          onClick={() => plan.mutate()}
        >
          <WrenchIcon />
          Plan Repairs
        </Button>
      }
    >
      {query.isError && !data ? (
        <ErrorNotice error={query.error} title="Could not load the repairs" />
      ) : !data ? (
        <Skeleton className="h-32" label="Loading the repairs" />
      ) : (
        <RepairBody summary={data} />
      )}
    </Panel>
  )
}

function RepairBody({ summary }: { summary: RepairSummary }) {
  const judged = summary.fixed + summary.notFixed + summary.errors
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric boxed size="sm" label="Near Misses Waiting" value={fmt.int(summary.waiting)} />
        <Metric boxed size="sm" label="Repairs Running" value={fmt.int(summary.pending)} />
        <Metric
          boxed
          size="sm"
          label="Fixed"
          value={fmt.int(summary.fixed)}
          tone={summary.fixed > 0 ? 'profit' : 'neutral'}
          hint={
            judged
              ? `${fmt.pct(summary.fixed / judged, 0)} of ${fmt.int(judged)} judged`
              : undefined
          }
        />
        <Metric
          boxed
          size="sm"
          label="Not Fixed"
          value={fmt.int(summary.notFixed + summary.errors)}
        />
      </div>
      {summary.moves.length === 0 ? (
        <p className="text-body-compact text-pretty text-ink-subtle">
          Nothing learned yet. Plan repairs, run the task from Tasks, and each fix's success rate
          appears here as its results are judged.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-body-compact">
            <thead>
              <tr className="text-left text-caption text-ink-subtle">
                <th className="py-1.5 pr-3 font-medium">Failure</th>
                <th className="py-1.5 pr-3 font-medium">Fix</th>
                <th className="py-1.5 pr-3 text-right font-medium">Fixed</th>
                <th className="py-1.5 text-right font-medium">Rate</th>
              </tr>
            </thead>
            <tbody>
              {summary.moves.slice(0, 12).map((m) => (
                <tr key={`${m.failure}:${m.move}`} className="border-t border-hairline">
                  <td className="py-1.5 pr-3 num text-ink-muted">{m.failure}</td>
                  <td className="py-1.5 pr-3 text-ink">{m.name}</td>
                  <td className="py-1.5 pr-3 text-right num">
                    {fmt.int(m.wins)} / {fmt.int(m.tries)}
                  </td>
                  <td className="py-1.5 text-right num">
                    {m.tries ? fmt.pct(m.wins / m.tries, 0) : DASH}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
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
