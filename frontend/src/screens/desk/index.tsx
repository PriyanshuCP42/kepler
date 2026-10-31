/**
 * Submission Desk: the best unsubmitted Critic passes, one per family, each with what it would
 * do to your combined performance, its Production Correlation, and the pyramid it counts
 * towards. Submitting stays a click on BRAIN, from the Alpha page.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate } from '@tanstack/react-router'
import {
  ChevronDownIcon,
  ChevronRightIcon,
  EllipsisIcon,
  ExternalLinkIcon,
  InfoIcon,
  MicroscopeIcon,
  RefreshCwIcon,
  ShieldCheckIcon,
  WrenchIcon,
} from 'lucide-react'
import { Fragment, useState } from 'react'
import { toast } from 'sonner'
import { errorMessage } from '@/api/http'
import { cn } from '@/lib/cn'
import { DASH, fmt, isNum } from '@/lib/format'
import {
  type Candidate,
  type CheckLine,
  DESK_KEY,
  desk,
  type Effect,
  type Figures,
  type GuardView,
  type Readiness,
  type SubmissionCheck,
  type YearSharpe,
} from '@/screens/desk/api'
import {
  Button,
  Empty,
  ErrorNotice,
  LINK,
  Page,
  PageHeader,
  Panel,
  Segmented,
  Skeleton,
  TEXT_TONE,
  type Tone,
} from '@/ui/kit'
import { Menu, Tooltip } from '@/ui/overlay'

const POLL_MS = 60_000
const WHOLE = 'all'
const POWER = 'Power Pool'
/** Genius's two performance bars, by level, as your Genius page shows them. */
const GENIUS_BARS = [
  ['Expert', 0.5],
  ['Master', 1],
  ['Grandmaster', 2],
] as const

const READINESS: Record<Readiness, { label: string; tone: Tone; note: string }> = {
  ready: {
    label: 'Ready',
    tone: 'profit',
    note: 'Passes every test here and lifts your pool. Run Check Submission as the last step.',
  },
  check: {
    label: 'Check First',
    tone: 'warn',
    note: 'Lifts your pool; Production Correlation or the neighbour check still to run.',
  },
  hurts: {
    label: 'Lowers Pool',
    tone: 'muted',
    note: 'Would lower the combined Sharpe of your pool or its partition.',
  },
  fading: {
    label: 'Fading',
    tone: 'loss',
    note: 'Weak in a recent year, or its recent years keep too little of its Sharpe.',
  },
  fails_check: {
    label: 'Fails Check',
    tone: 'loss',
    note: "BRAIN's own submission check refuses it. Repair it, or write its Power Pool description.",
  },
  overfit: {
    label: 'Overfit',
    tone: 'loss',
    note: 'Neighbours fall away, too few trading days, or a test Sharpe luck explains.',
  },
  blocked: {
    label: 'Blocked',
    tone: 'loss',
    note: "Over BRAIN's correlation limit.",
  },
}

export function DeskScreen() {
  const queryClient = useQueryClient()
  const query = useQuery({ queryKey: DESK_KEY, queryFn: desk.get, refetchInterval: POLL_MS })
  const check = useMutation({
    mutationFn: desk.checkProd,
    onSuccess: () =>
      toast.success('Asking BRAIN for Production Correlation', {
        description: 'BRAIN limits these per hour, so it runs in the background. No quota used.',
      }),
    onError: (error) => toast.error(errorMessage(error)),
  })
  const navigate = useNavigate()
  const guard = useMutation({
    mutationFn: () => desk.guard(),
    onSuccess: (task) => {
      toast.success(`Neighbour check written for ${task.finalists} candidates`, {
        description: 'Run it from Tasks. Each candidate is re-simulated with small changes.',
        action: { label: 'Open Tasks', onClick: () => void navigate({ to: '/tasks' }) },
      })
      void queryClient.invalidateQueries({ queryKey: DESK_KEY })
    },
    onError: (error) => toast.error(errorMessage(error)),
  })
  const submissionCheck = useMutation({
    mutationFn: desk.submissionCheck,
    onSuccess: () =>
      toast.success("Running BRAIN's submission check", {
        description: 'The same check as Check Submission, on each candidate, in the background.',
      }),
    onError: (error) => toast.error(errorMessage(error)),
  })
  const repairFailing = useMutation({
    mutationFn: () => desk.repairFailing(),
    onSuccess: (task) => {
      toast.success(`Repairs written for ${task.alphas} fixable alphas`, {
        description: 'Aimed at the checks it refused. Run the task from Tasks.',
        action: { label: 'Open Tasks', onClick: () => void navigate({ to: '/tasks' }) },
      })
    },
    onError: (error) => toast.error(errorMessage(error)),
  })
  const data = query.data
  const unchecked =
    data?.candidates.filter(
      (c) =>
        c.submission?.status === 'not_run' && (c.readiness === 'ready' || c.readiness === 'check'),
    ).length ?? 0
  const fixable = data?.candidates.filter((c) => c.fix?.action === 'repair').length ?? 0
  const unguarded =
    data?.candidates.filter(
      (c) => c.guard.status === 'untested' && (c.readiness === 'ready' || c.readiness === 'check'),
    ).length ?? 0
  const missing = data?.candidates.filter((c) => c.prodCorrelation === null).length ?? 0

  return (
    <Page>
      <PageHeader
        title="Submission Desk"
        description="What each shortlisted alpha would do to your combined performance if you submitted it, best first."
        actions={
          <>
            <Button
              variant="primary"
              loading={submissionCheck.isPending}
              disabled={!unchecked}
              onClick={() => submissionCheck.mutate()}
              title="BRAIN's own Check Submission on each Ready or Check First candidate, in the background: no quota, no LLM"
            >
              <ShieldCheckIcon />
              Run Submission Check{unchecked ? ` (${unchecked})` : ''}
            </Button>
            <Menu
              trigger={
                <Button aria-label="More actions">
                  <EllipsisIcon />
                  More
                </Button>
              }
              items={[
                {
                  label: `Repair fixable alphas${fixable ? ` (${fixable})` : ''}`,
                  icon: <WrenchIcon />,
                  disabled: !fixable || repairFailing.isPending,
                  onClick: () => repairFailing.mutate(),
                },
                {
                  label: `Run neighbour check${unguarded ? ` (${Math.min(unguarded, 20)})` : ''}`,
                  icon: <MicroscopeIcon />,
                  disabled: !unguarded || guard.isPending,
                  onClick: () => guard.mutate(),
                },
                {
                  label: `Check production correlation${missing ? ` (${missing})` : ''}`,
                  icon: <ShieldCheckIcon />,
                  disabled: !missing || check.isPending,
                  onClick: () => check.mutate(),
                },
                {
                  label: 'Refresh',
                  icon: <RefreshCwIcon />,
                  onClick: () => void queryClient.invalidateQueries({ queryKey: DESK_KEY }),
                },
              ]}
            />
          </>
        }
      />

      {query.isError && !data ? (
        <ErrorNotice error={query.error} title="Could not load the Submission Desk" />
      ) : !data ? (
        <Skeleton className="h-96" label="Weighing candidates" />
      ) : (
        <>
          <div className="grid gap-4 xl:grid-cols-[3fr_2fr]">
            <ReadyImpact
              impact={data.readyImpact}
              ready={data.readyIds}
              span={`${data.recentFrom ?? 2021}–${data.recentTo ?? 2023}`}
            />
            <PoolYears years={data.poolYears} from={data.recentFrom} to={data.recentTo} />
          </div>
          <CandidatesPanel
            candidates={data.candidates}
            span={`${data.recentFrom ?? 2021}–${data.recentTo ?? 2023}`}
            submitted={data.submitted}
          />
        </>
      )}
    </Page>
  )
}

function effectOf(c: Candidate, whole: boolean): Effect | undefined {
  return c.effects.find((e) => (whole ? e.scope === WHOLE : e.scope !== WHOLE && e.scope !== POWER))
}

function ReadyImpact({
  impact,
  ready,
  span,
}: {
  impact: Effect[] | undefined
  ready: string[] | undefined
  span: string
}) {
  if (!impact?.length) return null
  const rows: [string, string, Effect | undefined][] = [
    [
      'Combined Alpha Performance',
      'All your submitted alphas',
      impact.find((e) => e.scope === WHOLE),
    ],
    [
      'Combined Power Pool Alpha Performance',
      'Your Power Pool alphas',
      impact.find((e) => e.scope === POWER),
    ],
  ]
  const count = ready?.length ?? 0
  return (
    <Panel
      title={count ? `If You Submit All ${count} Ready` : 'Your Genius Performance Bars'}
      actions={
        <Tooltip
          content={`Combined performance: Sharpe, Fitness and margin up, turnover and drawdown down, over ${span}, the nearest years to the hidden ones BRAIN scores (from 2024). BRAIN's own figure is the one that counts; the Genius Tracker compares the two. Bars: ${GENIUS_BARS.map(([name, bar]) => `${name} ${bar}`).join(', ')}.`}
        >
          <button
            type="button"
            className="text-ink-subtle hover:text-ink"
            aria-label="How this is measured"
          >
            <InfoIcon className="size-4" />
          </button>
        </Tooltip>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2">
        {rows.map(([title, who, effect]) => (
          <div key={title} className="flex flex-col gap-1.5 rounded-sm bg-surface-1 px-3 py-2.5">
            <div className="text-caption text-ink-subtle">
              {title} · {who}, {fmt.int(effect?.members ?? 0)}
            </div>
            <Score effect={effect} />
            <Parts impact={effect?.impact} />
          </div>
        ))}
      </div>
      {count > 0 && <p className="mt-2 text-caption text-ink-subtle">{ready?.join(' · ')}</p>}
    </Panel>
  )
}

function PoolYears({
  years,
  from,
  to,
}: {
  years: YearSharpe[] | undefined
  from: number
  to: number
}) {
  // Absent from a backend older than this screen, until it is restarted.
  if (!years?.length) return null
  const first = years[0]
  const last = years[years.length - 1]
  return (
    <Panel
      title="Your Pool, Year by Year"
      actions={
        <Tooltip
          content={`Combined Sharpe of your submitted alphas in each backtest year. BRAIN scores the hidden days after 2023, so ${from}–${to} (highlighted) is the nearest Kepler can see, and what candidates are judged on.`}
        >
          <button
            type="button"
            className="text-ink-subtle hover:text-ink"
            aria-label="What this shows"
          >
            <InfoIcon className="size-4" />
          </button>
        </Tooltip>
      }
    >
      <div className="flex flex-wrap gap-1.5">
        {years.map((y) => {
          const recent = y.year >= from && y.year <= to
          return (
            <div
              key={y.year}
              className={cn(
                'flex min-w-12 flex-col items-center rounded-sm px-1.5 py-1',
                recent ? 'bg-surface-3' : 'bg-surface-1',
              )}
            >
              <span className="num text-caption text-ink-subtle">{y.year}</span>
              <span className="num text-body text-ink">{fmt.ratio(y.sharpe, 1)}</span>
            </div>
          )
        })}
      </div>
      {isNum(first?.sharpe) && isNum(last?.sharpe) && last.sharpe < first.sharpe && (
        <p className="mt-2 text-caption text-pretty text-ink-subtle">
          Falling from {fmt.ratio(first.sharpe, 1)} to {fmt.ratio(last.sharpe, 1)}: only alphas
          strong in every year of {from}–{to} reach Ready.
        </p>
      )}
    </Panel>
  )
}

const PART_LABELS = {
  sharpe: 'Sharpe',
  fitness: 'Fitness',
  margin: 'Margin',
  turnover: 'Turnover',
  drawdown: 'Drawdown',
} as const

const percent = (v: number) => `${v > 0 ? '+' : ''}${fmt.pct(v, 1)}`

/** The combined-performance change, and under it how the Sharpe moved. */
function Score({ effect }: { effect: Effect | undefined }) {
  const impact = effect?.impact
  if (!effect) return <span className="text-ink-subtle">{DASH}</span>
  if (!impact) return <Change effect={effect} recent />
  return (
    <span className="flex flex-col">
      <span
        className={cn(
          'num text-body font-medium',
          TEXT_TONE[impact.total >= 0 ? 'profit' : 'loss'],
        )}
        title={`Combined performance over ${impact.window}: Sharpe, Fitness, margin, turnover and drawdown together`}
      >
        {percent(impact.total)}
      </span>
      <span className="text-caption text-ink-subtle">
        Sharpe <Change effect={effect} recent />
      </span>
    </span>
  )
}

/** Each measure's part of the change, better always positive. */
function Parts({ impact }: { impact: Effect['impact'] }) {
  if (!impact) return null
  return (
    <span className="flex flex-wrap gap-x-3 gap-y-0.5 text-caption">
      {(Object.keys(PART_LABELS) as (keyof typeof PART_LABELS)[]).map((key) => {
        const v = impact.parts[key]
        if (!isNum(v)) return null
        return (
          <span key={key} className="whitespace-nowrap">
            <span className="text-ink-subtle">{PART_LABELS[key]} </span>
            <span className={cn('num', TEXT_TONE[v >= 0 ? 'profit' : 'loss'])}>{percent(v)}</span>
          </span>
        )
      })}
    </span>
  )
}

function Change({ effect, recent = false }: { effect: Effect | undefined; recent?: boolean }) {
  const shown = recent ? effect?.recentAfter : effect?.after
  const after = shown?.sharpe
  if (!effect || !isNum(after)) return <span className="text-ink-subtle">{DASH}</span>
  const before = (recent ? effect.recentBefore : effect.before)?.sharpe
  if (!isNum(before)) {
    return (
      <span className="num text-ink" title="None of your alphas is in this partition yet">
        new · {fmt.ratio(after)}
      </span>
    )
  }
  const change = after - before
  return (
    <span className="num whitespace-nowrap">
      <span className="text-ink-muted">{fmt.ratio(before)}</span>
      <span className="text-ink-subtle"> → </span>
      <span className="text-ink">{fmt.ratio(after)}</span>{' '}
      <span className={TEXT_TONE[change >= 0 ? 'profit' : 'loss']}>
        ({change >= 0 ? '+' : ''}
        {fmt.ratio(change)})
      </span>
    </span>
  )
}

type View = 'worth' | 'ready' | 'refused' | 'aside' | 'all'

const VIEWS: Record<View, { label: string; keep: (c: Candidate) => boolean }> = {
  worth: {
    label: 'Worth a look',
    keep: (c) => ['ready', 'check', 'fails_check'].includes(c.readiness),
  },
  ready: { label: 'Ready', keep: (c) => c.readiness === 'ready' },
  refused: { label: 'Fails Check', keep: (c) => c.readiness === 'fails_check' },
  aside: {
    label: 'Set aside',
    keep: (c) => ['hurts', 'fading', 'overfit', 'blocked'].includes(c.readiness),
  },
  all: { label: 'All', keep: () => true },
}

function CandidatesPanel({
  candidates,
  span,
  submitted,
}: {
  candidates: Candidate[]
  span: string
  submitted: number
}) {
  const [open, setOpen] = useState<string | null>(null)
  const [view, setView] = useState<View>('worth')
  const shown = candidates.filter(VIEWS[view].keep)
  return (
    <Panel
      title="Candidates"
      description={`Best first, against your ${fmt.int(submitted)} submitted alphas. Click a label for what to do, a row for every figure.`}
      actions={
        <Segmented
          label="Show"
          value={view}
          onChange={setView}
          items={(Object.keys(VIEWS) as View[]).map((v) => ({
            value: v,
            label: `${VIEWS[v].label} ${candidates.filter(VIEWS[v].keep).length}`,
          }))}
        />
      }
      bodyClassName={shown.length ? 'p-0' : ''}
    >
      {candidates.length === 0 ? (
        <Empty title="No candidates yet">
          Candidates appear as the Critic passes alphas and the Family Clerk places them in new
          families.
        </Empty>
      ) : shown.length === 0 ? (
        <Empty title={`Nothing in ${VIEWS[view].label}`}>Pick another view above.</Empty>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-body">
            <thead>
              <tr className="border-b border-hairline text-left text-caption uppercase tracking-wide text-ink-subtle">
                <th className="px-4 py-2.5 font-medium">Alpha</th>
                <th className="px-4 py-2.5 font-medium">Verdict</th>
                <th className="px-4 py-2.5 font-medium">Market</th>
                <th className="px-4 py-2.5 font-medium">Your pool {span}</th>
                <th className="px-4 py-2.5 font-medium">Power Pool</th>
                <th className="px-4 py-2.5 font-medium">Checks</th>
                <th className="px-4 py-2.5 font-medium">Pyramid</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((c) => {
                const expanded = open === c.alphaId
                return (
                  <Fragment key={c.alphaId}>
                    <tr
                      className={cn(
                        'cursor-pointer border-b border-hairline-subtle transition-colors hover:bg-surface-2',
                        expanded && 'bg-surface-1',
                      )}
                      onClick={(e) => {
                        // A label's menu, rendered elsewhere, still bubbles its clicks here.
                        if ((e.target as HTMLElement).closest('button, a, [role="menuitem"]'))
                          return
                        setOpen(expanded ? null : c.alphaId)
                      }}
                    >
                      <td className="px-4 py-3">
                        <span className="flex items-center gap-1.5">
                          {expanded ? (
                            <ChevronDownIcon className="size-3.5 text-ink-subtle" aria-hidden />
                          ) : (
                            <ChevronRightIcon className="size-3.5 text-ink-subtle" aria-hidden />
                          )}
                          <Link
                            to="/alpha/$alphaId"
                            params={{ alphaId: c.alphaId }}
                            className={cn('num', LINK)}
                            onClick={(e) => e.stopPropagation()}
                          >
                            {c.alphaId}
                          </Link>
                        </span>
                        {c.repairOf && (
                          <div className="ml-5 text-caption text-ink-subtle">
                            repair of {c.repairOf.origin} · {c.repairOf.name}
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <VerdictChip candidate={c} onDetails={() => setOpen(c.alphaId)} />
                      </td>
                      <td className="num px-4 py-3 text-ink-muted">
                        {[c.region, isNum(c.delay) ? `D${c.delay}` : null]
                          .filter(Boolean)
                          .join(' · ')}
                        <div className="text-caption text-ink-subtle">
                          {c.universe}
                          {c.powerPool && ' · PP'}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <Score effect={effectOf(c, true)} />
                      </td>
                      <td className="px-4 py-3">
                        {c.powerPool ? (
                          <Score effect={c.effects.find((e) => e.scope === POWER)} />
                        ) : (
                          <span className="text-caption text-ink-subtle">{DASH}</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex flex-col gap-0.5 text-body-compact">
                          <span>
                            <span className="text-caption text-ink-subtle">BRAIN </span>
                            <SubmissionBadge submission={c.submission} />
                          </span>
                          <span>
                            <span className="text-caption text-ink-subtle">Neighbours </span>
                            <GuardBadge guard={c.guard} />
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        {c.pyramid ? (
                          <span className="whitespace-nowrap">
                            <span className="num text-body-compact text-ink-muted">
                              {c.pyramid.name}
                            </span>
                            <div className="text-caption text-ink-subtle">
                              {c.pyramid.submitted}/3
                              {c.pyramid.completes && (
                                <span className={TEXT_TONE.profit}> · completes it</span>
                              )}
                            </div>
                          </span>
                        ) : (
                          <span className="text-ink-subtle">{DASH}</span>
                        )}
                      </td>
                    </tr>
                    {expanded && (
                      <tr className="border-b border-hairline-subtle bg-surface-1">
                        <td colSpan={7} className="px-6 py-4">
                          <Details candidate={c} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}

const FIX_LABELS: Record<NonNullable<Candidate['fix']>['action'], { label: string; tone: Tone }> = {
  describe: { label: 'Add description', tone: 'warn' },
  repair: { label: 'Repair', tone: 'warn' },
  drop: { label: 'Drop', tone: 'loss' },
  repaired: { label: 'Repaired', tone: 'muted' },
}

/** The verdict, or for a refused alpha what to do about it: hover for why, click to act. */
function VerdictChip({ candidate, onDetails }: { candidate: Candidate; onDetails: () => void }) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const repair = useMutation({
    mutationFn: () => desk.repairOne(candidate.alphaId),
    onSuccess: () => {
      toast.success(`One repair round written for ${candidate.alphaId}`, {
        description: 'Up to 3 fixes, each judged again from scratch. Run it from Tasks.',
        action: { label: 'Open Tasks', onClick: () => void navigate({ to: '/tasks' }) },
      })
      void queryClient.invalidateQueries({ queryKey: DESK_KEY })
    },
    onError: (error) => toast.error(errorMessage(error)),
  })
  const fix = candidate.readiness === 'fails_check' ? candidate.fix : null
  const shown = fix ? FIX_LABELS[fix.action] : READINESS[candidate.readiness]
  const why = fix?.reason ?? candidate.reasons[0] ?? READINESS[candidate.readiness].note
  const open = () =>
    void navigate({ to: '/alpha/$alphaId', params: { alphaId: candidate.alphaId } })
  return (
    <Tooltip content={why}>
      <span className="inline-flex">
        <Menu
          align="start"
          trigger={
            <button
              type="button"
              className={cn(
                'inline-flex h-7 items-center gap-1 whitespace-nowrap rounded-pill border border-hairline-strong px-2.5 text-body-compact font-medium transition-colors hover:bg-surface-3',
                TEXT_TONE[shown.tone],
              )}
            >
              {shown.label}
              <ChevronDownIcon className="size-3 opacity-70" aria-hidden />
            </button>
          }
          items={[
            ...(fix?.action === 'repair'
              ? [
                  {
                    label: 'Repair this alpha (one round)',
                    icon: <WrenchIcon />,
                    disabled: repair.isPending,
                    onClick: () => repair.mutate(),
                  },
                ]
              : []),
            {
              label: fix?.action === 'describe' ? 'Add its description' : 'Open its Alpha page',
              icon: <ExternalLinkIcon />,
              onClick: open,
            },
            { label: 'Show every figure', icon: <ChevronDownIcon />, onClick: onDetails },
          ]}
        />
      </span>
    </Tooltip>
  )
}

function SubmissionBadge({ submission }: { submission: SubmissionCheck | null | undefined }) {
  if (!submission) return <span className="text-ink-subtle">{DASH}</span>
  const shown = {
    passes: { label: 'Passes', tone: 'profit' as Tone },
    fails: { label: 'Fails', tone: 'loss' as Tone },
    not_run: { label: 'Not run', tone: 'muted' as Tone },
  }[submission.status]
  return (
    <span className="whitespace-nowrap">
      <span className={cn('text-body-compact', TEXT_TONE[shown.tone])}>{shown.label}</span>
      {submission.status === 'not_run' && submission.warned.length > 0 && (
        <span
          className={cn('ml-1.5 text-caption', TEXT_TONE.warn)}
          title={submission.warned.map(checkText).join('; ')}
        >
          {submission.warned.length} warned
        </span>
      )}
    </span>
  )
}

const checkText = (c: CheckLine) =>
  isNum(c.value) && isNum(c.limit) ? `${c.name} ${c.value} against ${c.limit}` : c.name

function SubmissionDetails({
  submission,
  alphaId,
}: {
  submission: SubmissionCheck | null | undefined
  alphaId: string
}) {
  if (!submission) return null
  const lines = submission.status === 'fails' ? submission.failing : submission.warned
  return (
    <div className="flex flex-col gap-1 text-body-compact">
      <span className="text-ink">
        BRAIN's submission check:{' '}
        {submission.status === 'passes'
          ? 'passes'
          : submission.status === 'fails'
            ? 'refuses it'
            : 'not run yet'}
        {submission.checkedAt && (
          <span className="text-ink-subtle"> · {fmt.ago(submission.checkedAt)}</span>
        )}
      </span>
      {lines.length > 0 && (
        <ul className="flex flex-col gap-0.5 text-ink-muted">
          {lines.map((c) => (
            <li
              key={c.name}
              className={submission.status === 'fails' ? TEXT_TONE.loss : TEXT_TONE.warn}
            >
              {c.result} · {checkText(c)}
            </li>
          ))}
        </ul>
      )}
      {submission.needsDescription && (
        <span className="text-ink-muted">
          BRAIN asks for a Power Pool description it does not have (100 characters or more). An
          alpha that misses a threshold is accepted only through the Power Pool route:{' '}
          <Link to="/alpha/$alphaId" params={{ alphaId }} className={LINK}>
            add one on its Alpha page
          </Link>
          , then check it again.
        </span>
      )}
    </div>
  )
}

const GUARD: Record<GuardView['status'], { label: string; tone: Tone }> = {
  untested: { label: 'Not run', tone: 'muted' },
  running: { label: 'Running', tone: 'warn' },
  robust: { label: 'Robust', tone: 'profit' },
  fragile: { label: 'Fragile', tone: 'loss' },
  inconclusive: { label: 'Inconclusive', tone: 'muted' },
}

function GuardBadge({ guard }: { guard: GuardView }) {
  const shown = GUARD[guard.status]
  const count = guard.neighbours.length
  return (
    <span className="whitespace-nowrap">
      <span className={cn('text-body-compact', TEXT_TONE[shown.tone])}>{shown.label}</span>
      {count > 0 && (
        <span className="num ml-1.5 text-caption text-ink-subtle">
          {guard.held}/{guard.measured || count}
        </span>
      )}
    </span>
  )
}

function GuardDetails({ guard, testSharpe }: { guard: GuardView; testSharpe: number | null }) {
  const luck = isNum(testSharpe) && testSharpe < guard.luckBar
  return (
    <div className="flex flex-col gap-2">
      <ul className="flex flex-col gap-0.5 text-body-compact text-ink-muted">
        {(guard.recentYears?.length ?? 0) > 0 && (
          <li>
            Sharpe by recent year:{' '}
            {(guard.recentYears ?? []).map((y, i) => (
              <span key={y.year}>
                {i > 0 && ' · '}
                {y.year}{' '}
                <span
                  className={cn(
                    'num',
                    isNum(y.sharpe) && y.sharpe < 0.5 ? TEXT_TONE.loss : 'text-ink',
                  )}
                >
                  {fmt.ratio(y.sharpe)}
                </span>
              </span>
            ))}
            ; together {fmt.ratio(guard.recentSharpe)} against {fmt.ratio(guard.fullSharpe)} over
            its whole history.
          </li>
        )}
        <li className={luck ? TEXT_TONE.loss : undefined}>
          {fmt.int(guard.siblings)} simulations of this idea in its region: chance alone takes the
          best of them to a test Sharpe of {fmt.ratio(guard.luckBar)}; this one has{' '}
          {fmt.ratio(testSharpe)}.
        </li>
        {guard.testDays !== null && (
          <li className={guard.testDays < 100 ? TEXT_TONE.loss : undefined}>
            Trades on {fmt.int(guard.testDays)} days of the test years.
          </li>
        )}
        {guard.years > 0 && (
          <li>
            Made money in {guard.positiveYears} of the {guard.years} years it traded in.
          </li>
        )}
        {guard.testWithoutSpike !== null && (
          <li className={guard.testWithoutSpike <= 0 ? TEXT_TONE.loss : undefined}>
            Test years without their best 10 days:{' '}
            {guard.testWithoutSpike > 0 ? 'still make money' : 'lose money'}.
          </li>
        )}
      </ul>
      {guard.neighbours.length > 0 && (
        <table className="w-full max-w-xl text-body-compact">
          <thead>
            <tr className="text-left text-caption text-ink-subtle">
              <th className="py-1 pr-3 font-medium">Neighbour</th>
              <th className="py-1 pr-3 text-right font-medium">Sharpe</th>
              <th className="py-1 pr-3 text-right font-medium">Test Sharpe</th>
              <th className="py-1 font-medium">Held</th>
            </tr>
          </thead>
          <tbody>
            {guard.neighbours.map((n) => (
              <tr key={n.kind} className="border-t border-hairline">
                <td className="py-1 pr-3 text-ink-muted">{n.name}</td>
                <td className="num py-1 pr-3 text-right">{fmt.ratio(n.sharpe)}</td>
                <td className="num py-1 pr-3 text-right">{fmt.ratio(n.testSharpe)}</td>
                <td className="py-1">
                  {n.state === 'queued' ? (
                    <span className="text-ink-subtle">waiting</span>
                  ) : n.state === 'failed' ? (
                    <span className="text-ink-subtle">did not run</span>
                  ) : n.holds ? (
                    <span className={TEXT_TONE.profit}>yes</span>
                  ) : (
                    <span className={TEXT_TONE.loss}>no</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

const ROWS: { key: keyof Figures; label: string; show: (v: number | null) => string }[] = [
  { key: 'sharpe', label: 'Sharpe', show: (v) => fmt.ratio(v) },
  { key: 'fitness', label: 'Fitness', show: (v) => fmt.ratio(v) },
  { key: 'returns', label: 'Returns', show: (v) => fmt.pct(v) },
  { key: 'drawdown', label: 'Drawdown', show: (v) => fmt.pct(v) },
  { key: 'turnover', label: 'Turnover', show: (v) => fmt.pct(v) },
  { key: 'margin', label: 'Margin', show: (v) => fmt.bps(v) },
]

function Details({ candidate }: { candidate: Candidate }) {
  const partition = effectOf(candidate, false)
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-body-compact">
        {(
          [
            ['Test Sharpe', fmt.ratio(candidate.testSharpe)],
            ['Pool correlation', fmt.ratio(candidate.poolCorrelation)],
            [
              'Production correlation',
              isNum(candidate.prodCorrelation)
                ? fmt.ratio(candidate.prodCorrelation)
                : 'not checked',
            ],
            ['Critic score', fmt.ratio(candidate.criticScore, 1)],
            [
              partition ? `Partition ${partition.scope}` : 'Partition',
              partition?.impact ? percent(partition.impact.total) : DASH,
            ],
          ] as const
        ).map(([label, value]) => (
          <span key={label}>
            <span className="text-ink-subtle">{label} </span>
            <span className="num text-ink">{value}</span>
          </span>
        ))}
      </div>
      {candidate.fix && (
        <p className="text-body-compact text-pretty text-ink">
          <span className={cn('font-medium', TEXT_TONE[FIX_LABELS[candidate.fix.action].tone])}>
            {FIX_LABELS[candidate.fix.action].label}:
          </span>{' '}
          {candidate.fix.reason}
        </p>
      )}
      <ul className="flex flex-col gap-0.5 text-body-compact text-ink-muted">
        {candidate.reasons.map((r) => (
          <li key={r}>{r}</li>
        ))}
        <li>
          Own figures: Sharpe {fmt.ratio(candidate.sharpe)}, train{' '}
          {fmt.ratio(candidate.trainSharpe)}, test {fmt.ratio(candidate.testSharpe)}, Fitness{' '}
          {fmt.ratio(candidate.fitness)}, Critic score {fmt.ratio(candidate.criticScore, 1)}.
        </li>
      </ul>
      <SubmissionDetails submission={candidate.submission} alphaId={candidate.alphaId} />
      <GuardDetails guard={candidate.guard} testSharpe={candidate.testSharpe} />
      {candidate.expression && (
        <code className="block overflow-x-auto rounded bg-surface-2 px-2 py-1.5 text-caption text-ink">
          {candidate.expression}
        </code>
      )}
      <div className="flex flex-col gap-4">
        {candidate.effects.map((effect) => (
          <table key={effect.scope} className="w-full text-body-compact">
            <thead>
              <tr className="text-left text-caption text-ink-subtle">
                <th className="py-1 pr-3 font-medium">
                  {effect.scope === WHOLE
                    ? 'Your whole pool'
                    : effect.scope === POWER
                      ? 'Your Power Pool alphas'
                      : effect.scope}{' '}
                  ({fmt.int(effect.members)})
                </th>
                <th className="py-1 pr-3 text-right font-medium">Recent now</th>
                <th className="py-1 pr-3 text-right font-medium">If submitted</th>
                <th className="py-1 pr-3 text-right font-medium">All years now</th>
                <th className="py-1 text-right font-medium">If submitted</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row) => (
                <tr key={row.key} className="border-t border-hairline">
                  <td className="py-1 pr-3 text-ink-muted">{row.label}</td>
                  <td className="num py-1 pr-3 text-right text-ink-muted">
                    {effect.recentBefore ? row.show(effect.recentBefore[row.key]) : DASH}
                  </td>
                  <td className="num py-1 pr-3 text-right text-ink">
                    {effect.recentAfter ? row.show(effect.recentAfter[row.key]) : DASH}
                  </td>
                  <td className="num py-1 pr-3 text-right text-ink-muted">
                    {effect.before ? row.show(effect.before[row.key]) : DASH}
                  </td>
                  <td className="num py-1 text-right text-ink">
                    {effect.after ? row.show(effect.after[row.key]) : DASH}
                  </td>
                </tr>
              ))}
              {effect.impact && (
                <tr className="border-t border-hairline">
                  <td className="py-1 pr-3 font-medium text-ink">Combined performance</td>
                  <td colSpan={4} className="py-1">
                    <span
                      className={cn(
                        'num mr-3 font-medium',
                        TEXT_TONE[effect.impact.total >= 0 ? 'profit' : 'loss'],
                      )}
                    >
                      {percent(effect.impact.total)}
                    </span>
                    <Parts impact={effect.impact} />
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        ))}
      </div>
    </div>
  )
}
