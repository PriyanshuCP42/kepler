/**
 * Genius Tracker: the figures your Genius page shows, typed in now and then, set against what
 * the Submission Desk's estimate said on the same day. BRAIN publishes them nowhere Kepler can
 * read, and scores them on days it hides, so this is how the estimate gets checked.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { PlusIcon, RefreshCwIcon, Trash2Icon } from 'lucide-react'
import { type FormEvent, useState } from 'react'
import { toast } from 'sonner'
import { errorMessage } from '@/api/http'
import { cn } from '@/lib/cn'
import { DASH, fmt, isNum } from '@/lib/format'
import {
  type Bar,
  GENIUS_KEY,
  type GeniusView,
  geniusApi,
  LEVEL_KEY,
  type LevelRow,
  OSMOSIS_KEY,
  type OsmosisScope,
} from '@/screens/genius/api'
import { PlaybookPanel } from '@/screens/playbook/insight'
import {
  Button,
  Empty,
  ErrorNotice,
  Field,
  Input,
  Notice,
  Page,
  PageHeader,
  Panel,
  Skeleton,
  TEXT_TONE,
} from '@/ui/kit'
import { Confirm } from '@/ui/overlay'

export function GeniusScreen() {
  const queryClient = useQueryClient()
  const query = useQuery({ queryKey: GENIUS_KEY, queryFn: geniusApi.get })
  const sync = useMutation({
    mutationFn: geniusApi.sync,
    onSuccess: (view) => {
      queryClient.setQueryData(GENIUS_KEY, view)
      toast.success('Synced from BRAIN')
    },
    onError: (error) => toast.error(errorMessage(error)),
  })
  const data = query.data

  return (
    <Page>
      <PageHeader
        title="Genius Tracker"
        description={
          data?.syncedAt
            ? `Your Genius figures, synced from BRAIN ${fmt.ago(data.syncedAt)} and every ${data.syncHours ?? 6} hours, against what the Submission Desk estimated on the same days.`
            : 'Your Genius figures, synced from BRAIN every few hours, against what the Submission Desk estimated on the same days.'
        }
        actions={
          <Button variant="primary" loading={sync.isPending} onClick={() => sync.mutate()}>
            <RefreshCwIcon />
            Sync from BRAIN
          </Button>
        }
      />
      {data?.syncError && (
        <p className={cn('text-body-compact', TEXT_TONE.loss)}>
          The last sync failed: {data.syncError}
        </p>
      )}
      {query.isError && !data ? (
        <ErrorNotice error={query.error} title="Could not load the Genius tracker" />
      ) : !data ? (
        <Skeleton className="h-96" label="Loading your readings" />
      ) : (
        <>
          <div className="grid gap-4 xl:grid-cols-2">
            <BarsPanel data={data} />
            <AddReading />
          </div>
          <LevelPanel />
          <OsmosisPanel />
          <AgreementPanel data={data} />
          <ReadingsPanel data={data} />
        </>
      )}
      <PlaybookPanel place="genius" />
    </Page>
  )
}

// ── Bars ────────────────────────────────────────────────────────────────────────────────

function next(bar: Bar, levels: string[]): string {
  const value = bar.value
  if (!isNum(value)) return 'No reading yet'
  const index = bar.bars.findIndex((b) => value < b)
  const target = bar.bars[index]
  if (target === undefined) return 'Every bar cleared'
  const need = target - value
  return `${target >= 5 ? fmt.int(need) : fmt.ratio(need)} to ${levels[index] ?? 'the next level'}`
}

function BarsPanel({ data }: { data: GeniusView }) {
  const latest = data.readings.at(-1)
  return (
    <Panel
      title="Your Genius Bars"
      description={
        latest
          ? `${latest.level ? `Level ${latest.level}` : 'Your level'}${isNum(latest.rank) ? ` · rank ${fmt.int(latest.rank)}` : ''} · from ${latest.source === 'brain' ? 'BRAIN' : 'your reading'} of ${fmt.date(latest.readOn)}. Every bar must clear a level for that level.`
          : 'Sync from BRAIN to see where you stand.'
      }
    >
      <table className="w-full text-body-compact">
        <thead>
          <tr className="text-left text-caption text-ink-subtle">
            <th className="py-1.5 pr-3 font-medium">Bar</th>
            <th className="py-1.5 pr-3 text-right font-medium">You</th>
            {data.levels.map((l) => (
              <th key={l} className="py-1.5 pr-3 text-right font-medium">
                {l}
              </th>
            ))}
            <th className="py-1.5 font-medium">Next</th>
          </tr>
        </thead>
        <tbody>
          {data.bars.map((bar) => (
            <tr key={bar.metric} className="border-t border-hairline">
              <td className="py-1.5 pr-3 text-ink">{bar.label}</td>
              <td
                className={cn(
                  'num py-1.5 pr-3 text-right',
                  TEXT_TONE[!isNum(bar.value) ? 'muted' : bar.level ? 'profit' : 'neutral'],
                )}
              >
                {isNum(bar.value)
                  ? fmt.ratio(
                      bar.value,
                      bar.metric.startsWith('s') || bar.metric.startsWith('py') ? 0 : 2,
                    )
                  : DASH}
              </td>
              {bar.bars.map((b, i) => (
                <td
                  key={data.levels[i]}
                  className={cn(
                    'num py-1.5 pr-3 text-right',
                    isNum(bar.value) && bar.value >= b ? TEXT_TONE.profit : 'text-ink-subtle',
                  )}
                >
                  {b}
                </td>
              ))}
              <td className="py-1.5 text-caption text-ink-subtle">{next(bar, data.levels)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  )
}

// ── Adding a reading ────────────────────────────────────────────────────────────────────

const today = () => new Date().toISOString().slice(0, 10)
const number = (text: string) => (text.trim() === '' ? null : Number(text))

function AddReading() {
  const queryClient = useQueryClient()
  const [form, setForm] = useState({
    readOn: today(),
    combined: '',
    powerPool: '',
    signals: '',
    pyramids: '',
    note: '',
  })
  const set = (key: keyof typeof form) => (e: { target: { value: string } }) =>
    setForm((f) => ({ ...f, [key]: e.target.value }))
  const add = useMutation({
    mutationFn: () =>
      geniusApi.add({
        readOn: form.readOn,
        combined: number(form.combined),
        powerPool: number(form.powerPool),
        signals: number(form.signals),
        pyramids: number(form.pyramids),
        note: form.note.trim() || null,
      }),
    onSuccess: (view) => {
      queryClient.setQueryData(GENIUS_KEY, view)
      setForm((f) => ({ ...f, combined: '', powerPool: '', signals: '', pyramids: '', note: '' }))
      toast.success('Reading saved')
    },
    onError: (error) => toast.error(errorMessage(error)),
  })
  const submit = (e: FormEvent) => {
    e.preventDefault()
    add.mutate()
  }

  return (
    <Panel
      title="Add a Reading by Hand"
      description="Only needed when the sync cannot reach BRAIN, or to note a figure from another day. Synced readings arrive by themselves every few hours."
    >
      <form className="grid grid-cols-2 gap-3" onSubmit={submit}>
        <Field label="Date read">
          <Input type="date" value={form.readOn} onChange={set('readOn')} required />
        </Field>
        <Field label="Combined Alpha Performance">
          <Input
            type="number"
            step="0.01"
            value={form.combined}
            onChange={set('combined')}
            placeholder="0.80"
          />
        </Field>
        <Field label="Combined Power Pool Performance">
          <Input
            type="number"
            step="0.01"
            value={form.powerPool}
            onChange={set('powerPool')}
            placeholder="0.60"
          />
        </Field>
        <Field label="Signals">
          <Input
            type="number"
            min="0"
            step="1"
            value={form.signals}
            onChange={set('signals')}
            placeholder="8"
          />
        </Field>
        <Field label="Pyramids completed">
          <Input
            type="number"
            min="0"
            step="1"
            value={form.pyramids}
            onChange={set('pyramids')}
            placeholder="1"
          />
        </Field>
        <Field label="Note">
          <Input value={form.note} onChange={set('note')} placeholder="Optional" maxLength={500} />
        </Field>
        <div className="col-span-2 flex justify-end">
          <Button type="submit" variant="primary" loading={add.isPending}>
            <PlusIcon />
            Save Reading
          </Button>
        </div>
      </form>
    </Panel>
  )
}

// ── How well the estimate tracks BRAIN ──────────────────────────────────────────────────

const signed = (v: number | null | undefined, digits = 2) =>
  isNum(v) ? `${v > 0 ? '+' : ''}${fmt.ratio(v, digits)}` : DASH

function AgreementPanel({ data }: { data: GeniusView }) {
  const a = data.agreement
  return (
    <Panel
      title="Does the Estimate Track BRAIN?"
      description="Between each pair of readings, the alphas you submitted moved BRAIN's Combined Alpha Performance and Kepler's estimate of it: Sharpe, Fitness, margin, turnover and drawdown together, as the Submission Desk ranks. When the two move together, the Desk's ranking can be trusted."
    >
      <div className="flex flex-col gap-3">
        <p className="text-body text-pretty text-ink">{a.verdict}</p>
        <p className="text-body-compact text-ink-muted">
          {a.pairs} step{a.pairs === 1 ? '' : 's'} measured · same direction in {a.sameDirection}
          {isNum(a.correlation) && <> · correlation {fmt.ratio(a.correlation)}</>}
        </p>
        {data.steps.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-body-compact">
              <thead>
                <tr className="text-left text-caption text-ink-subtle">
                  <th className="py-1.5 pr-3 font-medium">Between</th>
                  <th className="py-1.5 pr-3 text-right font-medium">Submitted</th>
                  <th className="py-1.5 pr-3 text-right font-medium">BRAIN Combined</th>
                  <th className="py-1.5 pr-3 text-right font-medium">Estimated Performance</th>
                  <th className="py-1.5 pr-3 text-right font-medium">Estimated Sharpe</th>
                  <th className="py-1.5 pr-3 text-right font-medium">BRAIN Power Pool</th>
                  <th className="py-1.5 text-right font-medium">Estimate</th>
                </tr>
              </thead>
              <tbody>
                {data.steps.map((s) => {
                  const move = isNum(s.estimateQuality) ? s.estimateQuality : s.estimateCombined
                  const agree =
                    isNum(s.brainCombined) && isNum(move) ? s.brainCombined * move > 0 : null
                  return (
                    <tr key={`${s.since}:${s.until}`} className="border-t border-hairline">
                      <td className="num py-1.5 pr-3 text-ink-muted">
                        {fmt.date(s.since)} → {fmt.date(s.until)}
                      </td>
                      <td className="num py-1.5 pr-3 text-right" title={s.submitted.join(', ')}>
                        {fmt.int(s.submitted.length)}
                      </td>
                      <td className="num py-1.5 pr-3 text-right text-ink">
                        {signed(s.brainCombined)}
                      </td>
                      <td
                        className={cn(
                          'num py-1.5 pr-3 text-right',
                          agree === null ? 'text-ink-muted' : TEXT_TONE[agree ? 'profit' : 'loss'],
                        )}
                        title="Sharpe, Fitness, margin, turnover and drawdown together"
                      >
                        {isNum(s.estimateQuality)
                          ? `${s.estimateQuality > 0 ? '+' : ''}${fmt.pct(s.estimateQuality, 1)}`
                          : DASH}
                      </td>
                      <td className="num py-1.5 pr-3 text-right text-ink-muted">
                        {signed(s.estimateCombined)}
                      </td>
                      <td className="num py-1.5 pr-3 text-right text-ink">
                        {signed(s.brainPowerPool)}
                      </td>
                      <td className="num py-1.5 text-right text-ink-muted">
                        {signed(s.estimatePowerPool)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Panel>
  )
}

// ── Readings ────────────────────────────────────────────────────────────────────────────

function ReadingsPanel({ data }: { data: GeniusView }) {
  const queryClient = useQueryClient()
  const remove = useMutation({
    mutationFn: geniusApi.remove,
    onSuccess: (view) => queryClient.setQueryData(GENIUS_KEY, view),
    onError: (error) => toast.error(errorMessage(error)),
  })
  const rows = [...data.readings].reverse()
  return (
    <Panel
      title="Readings"
      description={`Newest first. The estimate is the combined Sharpe over 2021–2023 of the alphas you had submitted by that day; today, with ${fmt.int(data.today?.alphas ?? 0)} submitted, it is ${fmt.ratio(data.today?.combined)}.`}
      bodyClassName={rows.length ? 'p-0' : ''}
    >
      {rows.length === 0 ? (
        <Empty title="No readings yet">
          Click Sync from BRAIN, or wait a minute after Kepler starts: it syncs by itself.
        </Empty>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-body">
            <thead>
              <tr className="border-b border-hairline text-left text-caption uppercase tracking-wide text-ink-subtle">
                <th className="px-4 py-2 font-medium">Date</th>
                <th className="px-4 py-2 font-medium">From</th>
                <th className="px-4 py-2 text-right font-medium">Combined</th>
                <th className="px-4 py-2 text-right font-medium">Estimate</th>
                <th className="px-4 py-2 text-right font-medium">Power Pool</th>
                <th className="px-4 py-2 text-right font-medium">Estimate</th>
                <th className="px-4 py-2 text-right font-medium">Signals</th>
                <th className="px-4 py-2 text-right font-medium">Pyramids</th>
                <th className="px-4 py-2 text-right font-medium">Submitted</th>
                <th className="px-4 py-2 font-medium">Note</th>
                <th className="px-4 py-2" />
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className="border-b border-hairline-subtle last:border-0">
                  <td className="num px-4 py-2 text-ink">{fmt.date(r.readOn)}</td>
                  <td className="px-4 py-2 text-body-compact text-ink-muted">
                    {r.source === 'brain' ? 'BRAIN' : 'You'}
                    {r.level && <span className="text-ink-subtle"> · {r.level}</span>}
                  </td>
                  <td className="num px-4 py-2 text-right text-ink">{fmt.ratio(r.combined)}</td>
                  <td className="num px-4 py-2 text-right text-ink-muted">
                    {fmt.ratio(r.estimateCombined)}
                  </td>
                  <td className="num px-4 py-2 text-right text-ink">{fmt.ratio(r.powerPool)}</td>
                  <td className="num px-4 py-2 text-right text-ink-muted">
                    {fmt.ratio(r.estimatePowerPool)}
                  </td>
                  <td className="num px-4 py-2 text-right">{fmt.int(r.signals)}</td>
                  <td className="num px-4 py-2 text-right">{fmt.int(r.pyramids)}</td>
                  <td className="num px-4 py-2 text-right text-ink-muted">{fmt.int(r.alphas)}</td>
                  <td className="px-4 py-2 text-body-compact text-ink-muted">{r.note}</td>
                  <td className="px-4 py-2 text-right">
                    <Button
                      size="sm"
                      variant="ghost"
                      aria-label={`Delete the reading of ${r.readOn}`}
                      loading={remove.isPending && remove.variables === r.id}
                      onClick={() => remove.mutate(r.id)}
                    >
                      <Trash2Icon />
                    </Button>
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

// ── Osmosis ─────────────────────────────────────────────────────────────────────────────

const scopeName = (s: OsmosisScope) => `${s.region}/${s.delay}`

function OsmosisPanel() {
  const queryClient = useQueryClient()
  const query = useQuery({ queryKey: OSMOSIS_KEY, queryFn: geniusApi.osmosis, retry: false })
  const [target, setTarget] = useState<OsmosisScope | null>(null)
  const apply = useMutation({
    mutationFn: (scope: OsmosisScope) => geniusApi.applyOsmosis([scopeName(scope)]),
    onSuccess: (done) => {
      queryClient.setQueryData(OSMOSIS_KEY, done.view)
      setTarget(null)
      if (done.failed.length) {
        toast.error(`${done.failed.length} Alphas were not updated on BRAIN`, {
          description: done.failed.join(', '),
        })
      } else {
        toast.success(`Osmosis points set on ${done.changed} Alphas`)
      }
    },
    onError: (error) => toast.error(errorMessage(error)),
  })
  const data = query.data
  const shown = data?.scopes.filter((s) => s.eligible || s.alphas >= 3) ?? []

  return (
    <Panel
      title="Osmosis Allocation"
      description={
        data
          ? `BRAIN needs ${data.minScopes} scopes, each exactly ${fmt.int(data.points)} points over at least ${data.minAlphas} Alphas. It locks every Sunday 23:59 US Eastern; next lock ${fmt.date(data.locksAt)}. The proposal keeps your strongest, least alike Alphas by their 2021–2023 Sharpe, which picked them, so read its Sharpe as in-sample.`
          : 'Your Osmosis points on BRAIN against an allocation that is valid and stronger.'
      }
      bodyClassName="flex flex-col gap-3"
    >
      {query.isError ? (
        <ErrorNotice error={query.error} title="Could not read your Osmosis points" />
      ) : !data ? (
        <Skeleton className="h-40" label="Reading your Osmosis points" />
      ) : (
        <>
          <p className="text-body-compact text-ink-muted">
            Valid scopes now:{' '}
            <span
              className={cn('num', TEXT_TONE[data.validNow >= data.minScopes ? 'profit' : 'loss'])}
            >
              {data.validNow} of {data.minScopes}
            </span>{' '}
            · scopes with enough Alphas: {data.eligible}
          </p>
          {data.advice.map((a) => (
            <Notice key={a} tone="warn" title={a} />
          ))}
          <div className="overflow-x-auto">
            <table className="w-full text-body-compact">
              <thead>
                <tr className="text-left text-caption text-ink-subtle">
                  <th className="py-1.5 pr-3 font-medium">Scope</th>
                  <th className="py-1.5 pr-3 text-right font-medium">Alphas</th>
                  <th className="py-1.5 pr-3 font-medium">On BRAIN now</th>
                  <th className="py-1.5 pr-3 text-right font-medium">Sharpe now</th>
                  <th className="py-1.5 pr-3 text-right font-medium">Proposed</th>
                  <th className="py-1.5 font-medium" />
                </tr>
              </thead>
              <tbody>
                {shown.map((s) => (
                  <tr key={scopeName(s)} className="border-t border-hairline">
                    <td className="num py-1.5 pr-3 text-ink">
                      {s.region} D{s.delay}
                    </td>
                    <td className="num py-1.5 pr-3 text-right">{fmt.int(s.alphas)}</td>
                    <td
                      className={cn(
                        'py-1.5 pr-3',
                        TEXT_TONE[s.currentlyValid ? 'profit' : s.currentPoints ? 'loss' : 'muted'],
                      )}
                    >
                      {s.currentPoints
                        ? `${fmt.int(s.currentPoints)} points on ${s.currentAlphas} · ${s.currentlyValid ? 'valid' : 'not valid'}`
                        : 'none'}
                    </td>
                    <td className="num py-1.5 pr-3 text-right text-ink-muted">
                      {fmt.ratio(s.currentSharpe)}
                    </td>
                    <td className="num py-1.5 pr-3 text-right text-ink">
                      {fmt.ratio(s.proposedSharpe)}
                    </td>
                    <td className="py-1.5 text-right">
                      {s.eligible && s.picks.length ? (
                        <Button size="sm" onClick={() => setTarget(s)}>
                          Review and apply
                        </Button>
                      ) : (
                        <span className="text-caption text-ink-subtle">
                          {s.shortOf} more Alpha{s.shortOf === 1 ? '' : 's'} to submit
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      <Confirm
        open={target !== null}
        onOpenChange={(open) => !open && !apply.isPending && setTarget(null)}
        title={target ? `Set Osmosis points for ${target.region} D${target.delay}?` : ''}
        confirmLabel="Set on BRAIN"
        cancelLabel="Not now"
        pending={apply.isPending}
        onConfirm={() => target && apply.mutate(target)}
      >
        {target && (
          <div className="flex flex-col gap-2">
            <p>
              Edits each Alpha whose points change on BRAIN, as its Submitted Alphas table does.
              Nothing is submitted. Points as of Sunday night are the ones BRAIN uses.
            </p>
            <div className="max-h-72 overflow-y-auto">
              <table className="w-full text-body-compact">
                <thead>
                  <tr className="text-left text-caption text-ink-subtle">
                    <th className="py-1 pr-3 font-medium">Alpha</th>
                    <th className="py-1 pr-3 text-right font-medium">Now</th>
                    <th className="py-1 pr-3 text-right font-medium">Proposed</th>
                    <th className="py-1 pr-3 text-right font-medium">2021–23 Sharpe</th>
                    <th className="py-1 font-medium">Why</th>
                  </tr>
                </thead>
                <tbody>
                  {target.picks.map((p) => (
                    <tr key={p.alphaId} className="border-t border-hairline">
                      <td className="num py-1 pr-3">{p.alphaId}</td>
                      <td className="num py-1 pr-3 text-right text-ink-muted">
                        {fmt.int(p.current)}
                      </td>
                      <td className="num py-1 pr-3 text-right">{fmt.int(p.points)}</td>
                      <td className="num py-1 pr-3 text-right">{fmt.ratio(p.recentSharpe)}</td>
                      <td className="py-1 text-caption text-ink-subtle">
                        {p.reason || (p.role === 'chosen' ? 'Strong and distinct' : '')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Confirm>
    </Panel>
  )
}

// ── Path to Master ──────────────────────────────────────────────────────────────────────

const figure = (row: LevelRow, v: number | null) =>
  !isNum(v) ? DASH : row.key === 'performance' || v % 1 ? fmt.ratio(v, 2) : fmt.int(v)

function LevelTable({ rows, bar }: { rows: LevelRow[]; bar: boolean }) {
  return (
    <table className="w-full text-body-compact">
      <thead>
        <tr className="text-left text-caption text-ink-subtle">
          <th className="py-1.5 pr-3 font-medium" />
          <th className="py-1.5 pr-3 text-right font-medium">You</th>
          {bar && <th className="py-1.5 pr-3 text-right font-medium">Bar</th>}
          <th className="py-1.5 text-right font-medium">Master median</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.key} className="border-t border-hairline" title={r.note || undefined}>
            <td className="py-1.5 pr-3 text-ink">
              {r.label}
              {r.lowerBetter && (
                <span className="text-caption text-ink-subtle"> · lower is better</span>
              )}
            </td>
            <td
              className={cn(
                'num py-1.5 pr-3 text-right',
                TEXT_TONE[r.met === null ? 'muted' : r.met ? 'profit' : 'loss'],
              )}
            >
              {figure(r, r.you)}
            </td>
            {bar && <td className="num py-1.5 pr-3 text-right text-ink">{figure(r, r.bar)}</td>}
            <td className="num py-1.5 text-right text-ink-muted">{figure(r, r.peers)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function LevelPanel() {
  const query = useQuery({ queryKey: LEVEL_KEY, queryFn: () => geniusApi.level(), retry: false })
  const plan = query.data
  return (
    <Panel
      title="Path to Master"
      description={
        plan
          ? `${plan.daysLeft} days to ${fmt.date(plan.quarterEnd)}. All three bars must clear; then the tie-breaker decides among everyone who cleared them. Master median: the ${plan.peersCounted} Master consultants on BRAIN's leaderboard this quarter.`
          : 'Your Genius figures this quarter against the Master bars and the consultants already there.'
      }
      bodyClassName="flex flex-col gap-4"
    >
      {query.isError ? (
        <ErrorNotice error={query.error} title="Could not read your Genius figures" />
      ) : !plan ? (
        <Skeleton className="h-40" label="Reading BRAIN's leaderboard" />
      ) : (
        <>
          <div className="grid gap-4 xl:grid-cols-2">
            <LevelTable rows={plan.bars} bar />
            <LevelTable rows={plan.tieBreakers} bar={false} />
          </div>
          <p className="text-caption text-ink-subtle">
            Combined performances:{' '}
            {Object.entries(plan.performances)
              .map(([k, v]) => `${k} ${fmt.ratio(v)}`)
              .join(' · ')}
            . The best one counts.
          </p>
          {plan.pace.map((p) => (
            <Notice key={p} tone="info" title={p} />
          ))}
        </>
      )}
    </Panel>
  )
}
