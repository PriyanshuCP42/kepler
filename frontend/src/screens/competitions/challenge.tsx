/**
 * A consultant challenge: its rules, where you stand on each of them today, and the
 * submittable Alphas that would lower your ranking metric most. Read from Kepler's own
 * records, so it still shows while the session is signed out.
 */

import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { useMemo } from 'react'
import type { ChallengeTab } from '@/api/core'
import { fmt } from '@/lib/format'
import { labTasks, type TaskAlpha } from '@/screens/tasks/api'
import { Badge, Empty, ErrorNotice, LINK, Metric, Notice, Panel, Skeleton } from '@/ui/kit'

const REASON: Record<string, string> = {
  OUTSIDE_WINDOW: 'outside the window',
  WRONG_REGION: 'not USA or GLB',
  PROD_CORRELATION: 'production correlation too high',
  PROD_NOT_MEASURED: 'production correlation not measured yet',
}

/** How many candidates to list: a day's worth of choices, not the whole vault. */
const CANDIDATES = 15

const regionOf = (a: TaskAlpha) => String(a.settings?.['region'] ?? '')

function Candidates({ challenge: c }: { challenge: ChallengeTab }) {
  const query = useQuery({ queryKey: ['submittable-alphas'], queryFn: labTasks.submittable })
  const counted = c.entries.filter((e) => e.eligible && e.prodCorrelation != null)
  const total = counted.reduce((sum, e) => sum + (e.prodCorrelation ?? 0), 0)
  const rows = useMemo(() => {
    const seen = new Set<string>()
    return (query.data ?? [])
      .filter((a) => {
        if (!a.alphaId || seen.has(a.alphaId)) return false
        seen.add(a.alphaId)
        return (
          a.submittable &&
          c.rules.regions.includes(regionOf(a)) &&
          a.prodCorrelation != null &&
          a.prodCorrelation < c.rules.maxProd
        )
      })
      .sort((a, b) => (a.prodCorrelation ?? 1) - (b.prodCorrelation ?? 1))
      .slice(0, CANDIDATES)
  }, [query.data, c.rules])

  return (
    <Panel
      title="Best Candidates for the Ranking"
      description={`Submittable ${c.rules.regions.join('/')} Alphas with a measured production correlation below ${fmt.ratio(c.rules.maxProd, 1)}, lowest first. "Average After" is your ranking metric if you submit that one next. Check each is not a lookalike of a recent submission.`}
    >
      {query.isError ? (
        <ErrorNotice error={query.error} title="Could not read the submittable Alphas" />
      ) : query.isPending ? (
        <Skeleton label="Reading submittable Alphas" />
      ) : rows.length === 0 ? (
        <Empty title="No measured candidate yet">
          Candidates appear once their production correlation is measured: run BRAIN's check on an
          Alpha from Submittable.
        </Empty>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-body-compact">
            <thead className="text-left text-ink-subtle">
              <tr className="border-b border-hairline">
                <th className="py-1.5 pr-3 font-medium">Alpha</th>
                <th className="py-1.5 pr-3 font-medium">Market</th>
                <th className="py-1.5 pr-3 text-right font-medium">Prod Corr</th>
                <th className="py-1.5 pr-3 text-right font-medium">Average After</th>
                <th className="py-1.5 pr-3 text-right font-medium">Sharpe</th>
                <th className="py-1.5 pr-3 text-right font-medium">Fitness</th>
                <th className="py-1.5 font-medium">Check</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((a) => (
                <tr key={a.alphaId ?? ''} className="border-b border-hairline last:border-0">
                  <td className="py-1.5 pr-3">
                    <Link
                      to="/alpha/$alphaId"
                      params={{ alphaId: a.alphaId ?? '' }}
                      className={LINK}
                    >
                      {a.alphaId}
                    </Link>
                  </td>
                  <td className="py-1.5 pr-3 text-ink-muted">
                    {regionOf(a)} {String(a.settings?.universe ?? '')}
                  </td>
                  <td className="num py-1.5 pr-3 text-right">{fmt.ratio(a.prodCorrelation)}</td>
                  <td className="num py-1.5 pr-3 text-right">
                    {fmt.ratio((total + (a.prodCorrelation ?? 0)) / (counted.length + 1))}
                  </td>
                  <td className="num py-1.5 pr-3 text-right">{fmt.ratio(a.sharpe)}</td>
                  <td className="num py-1.5 pr-3 text-right">{fmt.ratio(a.fitness)}</td>
                  <td className="py-1.5">
                    {a.pending ? (
                      <Badge tone="warn">Needs BRAIN check</Badge>
                    ) : (
                      <Badge tone="profit">Checked</Badge>
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

function Days({ challenge: c }: { challenge: ChallengeTab }) {
  if (!c.days.length) return <Empty title="The challenge has not started yet." />
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-body-compact">
        <thead className="text-left text-ink-subtle">
          <tr className="border-b border-hairline">
            <th className="py-1.5 pr-3 font-medium">BRAIN Day</th>
            <th className="py-1.5 pr-3 text-right font-medium">Simulations</th>
            <th className="py-1.5 pr-3 text-right font-medium">Submitted</th>
            <th className="py-1.5 pr-3 text-right font-medium">Eligible</th>
            <th className="py-1.5 font-medium">Daily Rules</th>
          </tr>
        </thead>
        <tbody>
          {c.days.map((d) => (
            <tr key={d.day} className="border-b border-hairline last:border-0">
              <td className="py-1.5 pr-3">{fmt.date(d.day)}</td>
              <td
                className={`num py-1.5 pr-3 text-right ${d.simulations < c.rules.minDailySimulations ? 'text-status-warning' : ''}`}
              >
                {fmt.int(d.simulations)}
              </td>
              <td className="num py-1.5 pr-3 text-right">{fmt.int(d.submissions)}</td>
              <td className="num py-1.5 pr-3 text-right">{fmt.int(d.eligible)}</td>
              <td className="py-1.5">
                {d.met === null ? (
                  <Badge tone="outline">Today</Badge>
                ) : d.met ? (
                  <Badge tone="profit">Met</Badge>
                ) : (
                  <Badge tone="loss">Missed</Badge>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Entries({ challenge: c }: { challenge: ChallengeTab }) {
  if (!c.entries.length) {
    return <Empty title="Nothing submitted in the window yet." />
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-body-compact">
        <thead className="text-left text-ink-subtle">
          <tr className="border-b border-hairline">
            <th className="py-1.5 pr-3 font-medium">Alpha</th>
            <th className="py-1.5 pr-3 font-medium">Region</th>
            <th className="py-1.5 pr-3 font-medium">BRAIN Day</th>
            <th className="py-1.5 pr-3 text-right font-medium">Prod Corr</th>
            <th className="py-1.5 pr-3 text-right font-medium">Sharpe</th>
            <th className="py-1.5 font-medium">Counts</th>
          </tr>
        </thead>
        <tbody>
          {c.entries.map((e) => (
            <tr key={e.alphaId} className="border-b border-hairline last:border-0">
              <td className="py-1.5 pr-3">
                <Link to="/alpha/$alphaId" params={{ alphaId: e.alphaId }} className={LINK}>
                  {e.alphaId}
                </Link>
                {e.themeMatch && (
                  <Badge tone="outline" className="ml-2">
                    GLB theme
                  </Badge>
                )}
              </td>
              <td className="py-1.5 pr-3 text-ink-muted">{e.region}</td>
              <td className="py-1.5 pr-3 text-ink-muted">{fmt.date(e.day)}</td>
              <td className="num py-1.5 pr-3 text-right">{fmt.ratio(e.prodCorrelation)}</td>
              <td className="num py-1.5 pr-3 text-right">{fmt.ratio(e.sharpe)}</td>
              <td className="py-1.5">
                {e.eligible ? (
                  <Badge tone="profit">Counts</Badge>
                ) : (
                  <Badge tone="loss">{e.reasons.map((r) => REASON[r] ?? r).join(', ')}</Badge>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function ChallengeTabBody({ challenge: c }: { challenge: ChallengeTab }) {
  const r = c.rules
  const open = c.started && !c.ended
  return (
    <>
      {c.warnings.map((w) => (
        <Notice key={w} tone="warn">
          {w}
        </Notice>
      ))}
      <Panel
        title={c.name}
        description={`${fmt.date(c.startDate)} – ${fmt.date(c.endDate)}. Ranked by ${r.ranking.toLowerCase()}; ties go to ${r.tieBreaker.toLowerCase()}. Prizes ${r.prizes.join(' / ')}.`}
        actions={
          <Badge tone={c.missedDays.length ? 'loss' : 'profit'}>
            {c.missedDays.length ? 'Daily rule missed' : 'On track'}
          </Badge>
        }
      >
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Metric
            boxed
            label="Average Prod Corr (Rank)"
            value={fmt.ratio(c.averageProd)}
            hint="Lower ranks higher"
          />
          <Metric
            boxed
            label="Eligible Alphas"
            value={`${fmt.int(c.eligible)} of ${fmt.int(r.minAlphas)}`}
            tone={c.eligible >= r.minAlphas ? 'profit' : 'warn'}
            hint={open ? `${fmt.int(c.alphasPerDay)} a day to finish` : undefined}
          />
          <Metric
            boxed
            label={`${r.themeRegion ?? ''} Power Pool Theme`}
            value={`${fmt.int(c.themeAlphas)} of ${fmt.int(r.minThemeAlphas)}`}
            tone={c.themeAlphas >= r.minThemeAlphas ? 'profit' : 'warn'}
          />
          <Metric boxed label="Days Left" value={fmt.int(c.daysLeft)} />
          <Metric
            boxed
            label="Simulations Today"
            value={`${fmt.int(c.simulationsToday)} of ${fmt.int(r.minDailySimulations)}`}
            tone={c.simulationsNeededToday ? 'warn' : 'profit'}
          />
          <Metric
            boxed
            label="Submitted Today"
            value={c.submissionsNeededToday ? 'Not yet' : 'Done'}
            tone={c.submissionsNeededToday ? 'warn' : 'profit'}
          />
          <Metric boxed label="Not Yet Measured" value={fmt.int(c.unmeasured)} />
          <Metric
            boxed
            label="BRAIN Session"
            value={c.signedIn === false ? 'Signed out' : c.signedIn ? 'Signed in' : '—'}
            tone={c.signedIn === false ? 'loss' : 'neutral'}
          />
        </div>
      </Panel>
      {open && <Candidates challenge={c} />}
      <Panel
        title="Daily Rules"
        description={`Each BRAIN day (US Eastern) needs ${fmt.int(r.minDailySimulations)} simulations and ${fmt.int(r.minDailySubmissions)} ${r.regions.join('/')} submission.`}
      >
        <Days challenge={c} />
      </Panel>
      <Panel
        title="Your Entries"
        description={`Alphas submitted in the window. Only ${r.regions.join('/')} Alphas with production correlation below ${fmt.ratio(r.maxProd, 1)} count.`}
      >
        <Entries challenge={c} />
      </Panel>
    </>
  )
}
