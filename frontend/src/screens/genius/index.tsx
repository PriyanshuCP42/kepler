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
import { type Bar, GENIUS_KEY, type GeniusView, geniusApi } from '@/screens/genius/api'
import {
  Button,
  Empty,
  ErrorNotice,
  Field,
  Input,
  Page,
  PageHeader,
  Panel,
  Skeleton,
  TEXT_TONE,
} from '@/ui/kit'

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
          <AgreementPanel data={data} />
          <ReadingsPanel data={data} />
        </>
      )}
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
