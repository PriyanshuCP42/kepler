/**
 * Submission Desk: the best unsubmitted Critic passes, one per family, each with what it would
 * do to your combined performance, its Production Correlation, and the pyramid it counts
 * towards. Submitting stays a click on BRAIN, from the Alpha page.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { ChevronDownIcon, ChevronRightIcon, RefreshCwIcon, ShieldCheckIcon } from 'lucide-react'
import { Fragment, useState } from 'react'
import { toast } from 'sonner'
import { errorMessage } from '@/api/http'
import { cn } from '@/lib/cn'
import { DASH, fmt, isNum } from '@/lib/format'
import {
  type Candidate,
  DESK_KEY,
  desk,
  type Effect,
  type Figures,
  type Readiness,
} from '@/screens/desk/api'
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
  TEXT_TONE,
  type Tone,
} from '@/ui/kit'

const POLL_MS = 60_000
const WHOLE = 'all'

const READINESS: Record<Readiness, { label: string; tone: Tone; note: string }> = {
  ready: {
    label: 'Ready',
    tone: 'profit',
    note: 'Production Correlation checked and under the limit; lifts your combined Sharpe.',
  },
  check: {
    label: 'Check First',
    tone: 'warn',
    note: 'Lifts your combined Sharpe; Production Correlation not checked yet.',
  },
  hurts: {
    label: 'Lowers Pool',
    tone: 'muted',
    note: 'Would lower the combined Sharpe of your pool or its partition.',
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
  const data = query.data
  const missing = data?.candidates.filter((c) => c.prodCorrelation === null).length ?? 0

  return (
    <Page>
      <PageHeader
        title="Submission Desk"
        description="What each shortlisted alpha would do to your combined performance if you submitted it, best first."
        actions={
          <>
            <Button
              loading={check.isPending}
              disabled={!missing}
              onClick={() => check.mutate()}
              title="Production Correlation is measured by BRAIN against every user's alphas"
            >
              <ShieldCheckIcon />
              Check Production Correlation{missing ? ` (${missing})` : ''}
            </Button>
            <Button
              loading={query.isFetching}
              onClick={() => void queryClient.invalidateQueries({ queryKey: DESK_KEY })}
            >
              <RefreshCwIcon />
              Refresh
            </Button>
          </>
        }
      />

      {query.isError && !data ? (
        <ErrorNotice error={query.error} title="Could not load the Submission Desk" />
      ) : !data ? (
        <Skeleton className="h-96" label="Weighing candidates" />
      ) : (
        <>
          <Summary candidates={data.candidates} submitted={data.submitted} />
          <CandidatesPanel candidates={data.candidates} />
          <p className="text-body-compact text-pretty text-ink-subtle">
            Combined figures are worked out here from stored daily PnL, at equal weight over the
            alphas trading each day, as BRAIN combines a pool; per partition this matches BRAIN's
            own figure. Open an alpha for BRAIN's exact Performance Comparison, then submit it
            there. Production Correlation above {fmt.ratio(data.prodCeiling, 1)} or correlation with
            your pool above {fmt.ratio(data.selfCeiling, 1)} blocks it.
          </p>
        </>
      )}
    </Page>
  )
}

function Summary({ candidates, submitted }: { candidates: Candidate[]; submitted: number }) {
  const count = (r: Readiness) => candidates.filter((c) => c.readiness === r).length
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
      {(Object.keys(READINESS) as Readiness[]).map((r) => (
        <Metric
          key={r}
          boxed
          size="sm"
          label={READINESS[r].label}
          value={fmt.int(count(r))}
          tone={count(r) && r !== 'hurts' ? READINESS[r].tone : 'neutral'}
          hint={READINESS[r].note}
        />
      ))}
      <Metric
        boxed
        size="sm"
        label="Your Pool"
        value={fmt.int(submitted)}
        hint="Submitted alphas"
      />
    </div>
  )
}

function effectOf(c: Candidate, whole: boolean): Effect | undefined {
  return c.effects.find((e) => (e.scope === WHOLE) === whole)
}

function Change({ effect }: { effect: Effect | undefined }) {
  const after = effect?.after?.sharpe
  if (!effect || !isNum(after)) return <span className="text-ink-subtle">{DASH}</span>
  const before = effect.before?.sharpe
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

function CandidatesPanel({ candidates }: { candidates: Candidate[] }) {
  const [open, setOpen] = useState<string | null>(null)
  return (
    <Panel
      title="Candidates"
      description="Critic passes that lead their family and are not submitted. Click a row for every combined figure."
      bodyClassName={candidates.length ? 'p-0' : ''}
    >
      {candidates.length === 0 ? (
        <Empty title="No candidates yet">
          Candidates appear as the Critic passes alphas and the Family Clerk places them in new
          families.
        </Empty>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-body">
            <thead>
              <tr className="border-b border-hairline text-left text-caption uppercase tracking-wide text-ink-subtle">
                <th className="px-4 py-2 font-medium">Alpha</th>
                <th className="px-4 py-2 font-medium">Verdict</th>
                <th className="px-4 py-2 font-medium">Market</th>
                <th className="px-4 py-2 text-right font-medium">Test Sharpe</th>
                <th className="px-4 py-2 text-right font-medium">Pool Corr</th>
                <th className="px-4 py-2 text-right font-medium">Prod Corr</th>
                <th className="px-4 py-2 font-medium">Your Pool Sharpe</th>
                <th className="px-4 py-2 font-medium">Partition Sharpe</th>
                <th className="px-4 py-2 font-medium">Pyramid</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => {
                const expanded = open === c.alphaId
                const partition = effectOf(c, false)
                const verdict = READINESS[c.readiness]
                return (
                  <Fragment key={c.alphaId}>
                    <tr
                      className="cursor-pointer border-b border-hairline-subtle hover:bg-surface-2"
                      onClick={() => setOpen(expanded ? null : c.alphaId)}
                    >
                      <td className="px-4 py-2">
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
                      </td>
                      <td className="px-4 py-2" title={c.reasons.join(' ')}>
                        <Badge tone={verdict.tone}>{verdict.label}</Badge>
                      </td>
                      <td className="num px-4 py-2 text-ink-muted">
                        {[c.region, isNum(c.delay) ? `D${c.delay}` : null, c.universe]
                          .filter(Boolean)
                          .join(' · ') || DASH}
                        {c.powerPool && <span className="ml-1.5 text-caption">PP</span>}
                      </td>
                      <td className="num px-4 py-2 text-right text-ink">
                        {fmt.ratio(c.testSharpe)}
                      </td>
                      <td className="num px-4 py-2 text-right text-ink-muted">
                        {fmt.ratio(c.poolCorrelation)}
                      </td>
                      <td
                        className={cn(
                          'num px-4 py-2 text-right',
                          TEXT_TONE[
                            !isNum(c.prodCorrelation)
                              ? 'muted'
                              : c.readiness === 'blocked'
                                ? 'loss'
                                : 'neutral'
                          ],
                        )}
                      >
                        {isNum(c.prodCorrelation) ? fmt.ratio(c.prodCorrelation) : 'not checked'}
                      </td>
                      <td className="px-4 py-2">
                        <Change effect={effectOf(c, true)} />
                      </td>
                      <td className="px-4 py-2">
                        <Change effect={partition} />
                        {partition && (
                          <div className="text-caption text-ink-subtle">
                            {partition.scope} · {fmt.int(partition.members)} alphas
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-2">
                        {c.pyramid ? (
                          <span className="whitespace-nowrap">
                            <span className="num text-ink-muted">{c.pyramid.name}</span>{' '}
                            <span className="num text-ink-subtle">{c.pyramid.submitted}/3</span>
                            {c.pyramid.completes && (
                              <Badge tone="profit" className="ml-1.5">
                                Completes
                              </Badge>
                            )}
                          </span>
                        ) : (
                          <span className="text-ink-subtle">{DASH}</span>
                        )}
                      </td>
                    </tr>
                    {expanded && (
                      <tr className="border-b border-hairline-subtle bg-surface-1">
                        <td colSpan={9} className="px-4 py-3">
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

const ROWS: { key: keyof Figures; label: string; show: (v: number | null) => string }[] = [
  { key: 'sharpe', label: 'Sharpe', show: (v) => fmt.ratio(v) },
  { key: 'fitness', label: 'Fitness', show: (v) => fmt.ratio(v) },
  { key: 'returns', label: 'Returns', show: (v) => fmt.pct(v) },
  { key: 'drawdown', label: 'Drawdown', show: (v) => fmt.pct(v) },
  { key: 'turnover', label: 'Turnover', show: (v) => fmt.pct(v) },
  { key: 'margin', label: 'Margin', show: (v) => fmt.bps(v) },
]

function Details({ candidate }: { candidate: Candidate }) {
  return (
    <div className="flex flex-col gap-3">
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
      {candidate.expression && (
        <code className="block overflow-x-auto rounded bg-surface-2 px-2 py-1.5 text-caption text-ink">
          {candidate.expression}
        </code>
      )}
      <div className="grid gap-4 lg:grid-cols-2">
        {candidate.effects.map((effect) => (
          <table key={effect.scope} className="w-full text-body-compact">
            <thead>
              <tr className="text-left text-caption text-ink-subtle">
                <th className="py-1 pr-3 font-medium">
                  {effect.scope === WHOLE ? 'Your whole pool' : effect.scope} (
                  {fmt.int(effect.members)})
                </th>
                <th className="py-1 pr-3 text-right font-medium">Now</th>
                <th className="py-1 text-right font-medium">If submitted</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row) => (
                <tr key={row.key} className="border-t border-hairline">
                  <td className="py-1 pr-3 text-ink-muted">{row.label}</td>
                  <td className="num py-1 pr-3 text-right text-ink-muted">
                    {effect.before ? row.show(effect.before[row.key]) : DASH}
                  </td>
                  <td className="num py-1 text-right text-ink">
                    {effect.after ? row.show(effect.after[row.key]) : DASH}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ))}
      </div>
    </div>
  )
}
