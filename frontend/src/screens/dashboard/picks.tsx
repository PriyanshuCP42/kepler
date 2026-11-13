/**
 * Today's Picks: the submittable Alphas the Submit Gate calls Ready, the ones to check with
 * BRAIN first, and the near misses to repair, with every reason. And what the Daily Autopilot
 * last did. Kepler never submits; the Ready ones are yours to submit.
 */

import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import type { components } from '@/api/generated'
import { http } from '@/api/http'
import { fmt } from '@/lib/format'
import { Badge, Empty, ErrorNotice, LINK, Notice, Panel, Skeleton } from '@/ui/kit'

type Picks = components['schemas']['Picks']
type Pick = components['schemas']['Pick']
type AutopilotStatus = components['schemas']['AutopilotStatus']

const REASON: Record<string, string> = {
  REVERSION_COMPONENT: 'reversion component',
  PROD_CORRELATION: 'production correlation ≥ 0.7',
  PROD_TWIN: 'its PnL twin failed production correlation',
  SELF_CORRELATION: 'self correlation ≥ 0.7',
  LOOKALIKE: 'repeats a recent submission',
  LOWERS_COMBINED: 'lowers combined Sharpe',
  LOWERS_POWER_POOL: 'lowers Power Pool Sharpe',
  LOW_SHARPE: 'Sharpe under the bar',
  LOW_FITNESS: 'Fitness under the bar',
  LOW_MARGIN: 'margin under 5 bps',
  RETURN_TO_DRAWDOWN: 'returns under 1.5× drawdown',
  TEST_DECAY: 'test years keep too little',
  NOT_SUBMITTABLE: 'neither regular nor on theme',
  BRAIN_CHECK_NOT_RUN: "BRAIN's check not run",
  POOL_EFFECT_NOT_MEASURED: 'pool effect not measured',
}

const reason = (r: string) =>
  REASON[r] ?? (r.startsWith('BRAIN_') ? `BRAIN: ${r.slice(6).toLowerCase()}` : r)

const signed = (v: number | null | undefined) =>
  v == null ? '—' : `${v >= 0 ? '+' : ''}${fmt.ratio(v, 3)}`

function Ready({ rows }: { rows: Pick[] }) {
  if (!rows.length) {
    return (
      <Empty title="Nothing is ready yet">
        Alphas appear here once BRAIN's check has run and every test passes.
      </Empty>
    )
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-body-compact">
        <thead className="text-left text-ink-subtle">
          <tr className="border-b border-hairline">
            <th className="py-1.5 pr-3 font-medium">Alpha</th>
            <th className="py-1.5 pr-3 font-medium">Market</th>
            <th className="py-1.5 pr-3 text-right font-medium">Sharpe</th>
            <th className="py-1.5 pr-3 text-right font-medium">Fitness</th>
            <th className="py-1.5 pr-3 text-right font-medium">Returns</th>
            <th className="py-1.5 pr-3 text-right font-medium">Ret / DD</th>
            <th className="py-1.5 pr-3 text-right font-medium">Margin</th>
            <th className="py-1.5 pr-3 text-right font-medium">Prod Corr</th>
            <th className="py-1.5 pr-3 text-right font-medium">Combined</th>
            <th className="py-1.5 pr-3 text-right font-medium">Power Pool</th>
            <th className="py-1.5 font-medium">Counts As</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr key={p.alphaId} className="border-b border-hairline last:border-0">
              <td className="py-1.5 pr-3">
                <Link to="/alpha/$alphaId" params={{ alphaId: p.alphaId }} className={LINK}>
                  {p.alphaId}
                </Link>
              </td>
              <td className="py-1.5 pr-3 text-ink-muted">
                {p.region} D{p.delay} {p.universe}
              </td>
              <td className="num py-1.5 pr-3 text-right">{fmt.ratio(p.sharpe)}</td>
              <td className="num py-1.5 pr-3 text-right">{fmt.ratio(p.fitness)}</td>
              <td className="num py-1.5 pr-3 text-right">{fmt.pct(p.returns)}</td>
              <td className="num py-1.5 pr-3 text-right">{fmt.ratio(p.returnToDrawdown)}</td>
              <td className="num py-1.5 pr-3 text-right">
                {p.margin == null ? '—' : `${fmt.ratio(p.margin * 10_000, 1)} bps`}
              </td>
              <td className="num py-1.5 pr-3 text-right">{fmt.ratio(p.prodCorrelation)}</td>
              <td className="num py-1.5 pr-3 text-right text-pnl-positive-text">
                {signed(p.wholeDelta)}
              </td>
              <td className="num py-1.5 pr-3 text-right text-pnl-positive-text">
                {signed(p.powerDelta)}
              </td>
              <td className="py-1.5">
                {p.regular && <Badge tone="profit">Regular</Badge>}{' '}
                {p.powerPool && <Badge tone="outline">Power Pool</Badge>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Waiting({ title, rows }: { title: string; rows: Pick[] }) {
  if (!rows.length) return null
  return (
    <div className="flex flex-col gap-1.5">
      <h3 className="text-body-compact font-medium text-ink-muted">{title}</h3>
      <ul className="flex flex-col gap-1 text-body-compact">
        {rows.slice(0, 6).map((p) => (
          <li key={p.alphaId} className="flex flex-wrap items-baseline gap-x-2">
            <Link to="/alpha/$alphaId" params={{ alphaId: p.alphaId }} className={LINK}>
              {p.alphaId}
            </Link>
            <span className="text-ink-subtle">
              {p.region} · Sharpe {fmt.ratio(p.sharpe)} · Fitness {fmt.ratio(p.fitness)}
            </span>
            <span className="text-ink-muted">{p.reasons.map(reason).join(', ')}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function TodaysPicks() {
  const picks = useQuery({
    queryKey: ['autopilot', 'picks'],
    queryFn: () => http.get<Picks>('/api/autopilot/picks'),
    staleTime: 5 * 60 * 1000,
    refetchInterval: 10 * 60 * 1000,
  })
  const status = useQuery({
    queryKey: ['autopilot', 'status'],
    queryFn: () => http.get<AutopilotStatus>('/api/autopilot/status'),
    refetchInterval: 60 * 1000,
  })
  const p = picks.data
  const s = status.data
  const top = Object.entries(p?.rejectedBy ?? {})
    .slice(0, 3)
    .map(([r, n]) => `${reason(r)} (${n})`)
    .join(', ')
  return (
    <Panel
      title="Today's Picks"
      description="Submittable Alphas the Submit Gate calls Ready: BRAIN's check passed, production correlation under 0.7, margin above 5 bps, returns at least 1.5 times the drawdown, both your combined and Power Pool Sharpe up, no reversion, and no repeat of a recent submission. Kepler never submits; these are yours to submit."
      actions={
        p && (
          <Badge tone={p.ready.length ? 'profit' : 'neutral'}>
            {fmt.int(p.ready.length)} ready
          </Badge>
        )
      }
    >
      <div className="flex flex-col gap-4">
        {s && (
          <p className="text-body-compact text-ink-subtle">
            {s.enabled ? 'Autopilot on' : 'Autopilot off'}
            {s.last?.['note'] ? ` · ${String(s.last['note'])}` : ''}
          </p>
        )}
        {s?.alerts.slice(0, 2).map((a) => (
          <Notice key={`${a.kind}-${a.title}`} tone="warn" title={a.title}>
            {a.text}
          </Notice>
        ))}
        {picks.isError ? (
          <ErrorNotice error={picks.error} title="Today's Picks could not load" />
        ) : picks.isPending || !p ? (
          <Skeleton label="Measuring candidates against your pools" className="h-32" />
        ) : (
          <>
            <Ready rows={p.ready} />
            <Waiting title="Check with BRAIN first" rows={p.check} />
            <Waiting title="Near misses to repair" rows={p.repair} />
            <p className="text-body-compact text-ink-subtle">
              {fmt.int(p.measured)} measured · {fmt.int(p.rejected)} rejected
              {top ? `: mostly ${top}` : ''}
            </p>
          </>
        )}
      </div>
    </Panel>
  )
}
