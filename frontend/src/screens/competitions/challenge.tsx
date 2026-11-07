/**
 * A consultant challenge: its rules, where you stand on each of them today, and the
 * submittable Alphas that would lower your ranking metric most. Read from Kepler's own
 * records, so it still shows while the session is signed out.
 */

import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { type ChallengeCandidate, type ChallengeTab, competitions } from '@/api/core'
import { fmt } from '@/lib/format'
import { Badge, Empty, ErrorNotice, LINK, Metric, Notice, Panel, Skeleton } from '@/ui/kit'

const REASON: Record<string, string> = {
  OUTSIDE_WINDOW: 'outside the window',
  WRONG_REGION: 'not USA or GLB',
  PROD_CORRELATION: 'production correlation too high',
  PROD_NOT_MEASURED: 'production correlation not measured yet',
  OUTSIDE_POWER_POOL_LIMITS: 'over Power Pool limits',
  NEEDS_REGULAR_OR_THEME: 'not regular and off theme',
  LOWERS_A_POOL: 'lowers a pool',
  EFFECT_NOT_MEASURED: 'effect not measured',
}

/** A pool's Sharpe change, signed, or a dash when it was not measured. */
function Change({ effect }: { effect: ChallengeCandidate['whole'] }) {
  if (effect?.after == null) return <span className="text-ink-subtle">—</span>
  const change = effect.after - (effect.before ?? 0)
  return (
    <span className={change >= 0 ? 'text-pnl-positive-text' : 'text-pnl-negative-text'}>
      {change >= 0 ? '+' : ''}
      {fmt.ratio(change, 3)}
    </span>
  )
}

function Candidates({ challenge: c }: { challenge: ChallengeTab }) {
  const query = useQuery({
    queryKey: ['competitions', 'challenges', c.id, 'candidates'],
    queryFn: () => competitions.challengeCandidates(c.id),
  })
  const rows = query.data?.candidates ?? []
  const qualifying = rows.filter((r) => r.qualifies).length
  return (
    <Panel
      title="Best Candidates"
      description={`Submittable ${c.rules.regions.join('/')} Alphas under ${fmt.ratio(c.rules.maxProd, 1)} production correlation. They qualify within Power Pool's limits, as regular Alphas or ones matching the month's theme, raising both your combined and your Power Pool Sharpe. Qualifying ones come first, lowest production correlation first. Check each is not a lookalike of a recent submission.`}
      actions={
        query.data && (
          <Badge tone={qualifying ? 'profit' : 'warn'}>{fmt.int(qualifying)} qualify</Badge>
        )
      }
    >
      {query.isError ? (
        <ErrorNotice error={query.error} title="Could not measure the candidates" />
      ) : query.isPending ? (
        <Skeleton label="Measuring candidates against your pools" />
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
                <th className="py-1.5 pr-3 text-right font-medium">Ops / Fields</th>
                <th className="py-1.5 pr-3 text-right font-medium">PP Corr</th>
                <th className="py-1.5 pr-3 text-right font-medium">Combined Sharpe</th>
                <th className="py-1.5 pr-3 text-right font-medium">PP Sharpe</th>
                <th className="py-1.5 font-medium">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((a) => (
                <tr key={a.alphaId} className="border-b border-hairline last:border-0">
                  <td className="py-1.5 pr-3">
                    <Link to="/alpha/$alphaId" params={{ alphaId: a.alphaId }} className={LINK}>
                      {a.alphaId}
                    </Link>
                  </td>
                  <td className="py-1.5 pr-3 text-ink-muted">
                    {a.region} {a.universe}
                    {a.themeMatch ? ' · theme' : a.regular ? ' · regular' : ''}
                  </td>
                  <td className="num py-1.5 pr-3 text-right">{fmt.ratio(a.prodCorrelation)}</td>
                  <td className="num py-1.5 pr-3 text-right">{fmt.ratio(a.averageAfter)}</td>
                  <td className="num py-1.5 pr-3 text-right">
                    {fmt.int(a.operators)} / {fmt.int(a.fields)}
                  </td>
                  <td className="num py-1.5 pr-3 text-right">
                    {fmt.ratio(a.powerPoolCorrelation)}
                  </td>
                  <td className="num py-1.5 pr-3 text-right">
                    <Change effect={a.whole} />
                  </td>
                  <td className="num py-1.5 pr-3 text-right">
                    <Change effect={a.power} />
                  </td>
                  <td className="py-1.5">
                    {a.qualifies ? (
                      <Badge tone={a.pending ? 'warn' : 'profit'}>
                        {a.pending ? 'Qualifies · needs BRAIN check' : 'Qualifies'}
                      </Badge>
                    ) : (
                      <Badge tone="muted">{a.reasons.map((r) => REASON[r] ?? r).join(', ')}</Badge>
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
