/**
 * Competitions: one tab for each competition BRAIN is running and each Power Pool theme, with
 * its dates, where you stand, the Alphas you have entered, and what the Playbook says about
 * it. Joining and submitting both happen on BRAIN.
 */

import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from '@tanstack/react-router'
import { ExternalLinkIcon, TrophyIcon } from 'lucide-react'
import { type Competition, competitions } from '@/api/core'
import type { components } from '@/api/generated'
import { PLATFORM_COMPETITIONS_URL } from '@/api/platform'
import { fmt } from '@/lib/format'
import { ChallengeTabBody } from '@/screens/competitions/challenge'
import { PlaybookPanel } from '@/screens/playbook/insight'
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
  TabBar,
  TabLink,
} from '@/ui/kit'

type Entry = components['schemas']['Entry']
type ThemeTab = components['schemas']['ThemeTab']

const BRAIN_COMPETITION_URL = (id: string) =>
  `${PLATFORM_COMPETITIONS_URL}${encodeURIComponent(id)}`

const DAY_MS = 86_400_000

const daysLeft = (iso: string | null | undefined) =>
  iso ? Math.max(0, Math.ceil((new Date(iso).getTime() - Date.now()) / DAY_MS)) : null

/** The Playbook place for a competition: its id without the cycle digits (AQP202611 → aqp). */
const placeOf = (id: string) => id.replace(/\d+$/, '').toLowerCase()
const themePlace = (t: ThemeTab) => `${t.region}-d${t.delay}-theme`.toLowerCase()
const themeKey = (t: ThemeTab) => `theme-${t.id.toLowerCase()}`

const CHECK_LABEL: Record<string, string> = {
  LOW_GLB_AMER_SHARPE: 'Americas Sharpe above 1',
  SUBMITTED_BEFORE_THEME: 'submitted before the theme began',
}

function Entries({ entries, theme }: { entries: Entry[]; theme?: boolean }) {
  if (!entries.length) {
    return (
      <Empty title="No Alphas entered yet">
        {theme
          ? 'Power Pool Alphas in this market tagged PowerPoolSelected show here.'
          : 'Alphas tagged with this competition’s id show here.'}
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
            <th className="py-1.5 pr-3 font-medium">Submitted</th>
            {theme && <th className="py-1.5 font-medium">Theme</th>}
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => (
            <tr key={e.alphaId} className="border-b border-hairline last:border-0">
              <td className="py-1.5 pr-3">
                <Link to="/alpha/$alphaId" params={{ alphaId: e.alphaId }} className={LINK}>
                  {e.alphaId}
                </Link>
              </td>
              <td className="py-1.5 pr-3 text-ink-muted">
                {e.region} D{e.delay} {e.universe}
              </td>
              <td className="num py-1.5 pr-3 text-right">{fmt.ratio(e.sharpe)}</td>
              <td className="num py-1.5 pr-3 text-right">{fmt.ratio(e.fitness)}</td>
              <td className="py-1.5 pr-3 text-ink-muted">{fmt.date(e.dateSubmitted)}</td>
              {theme && (
                <td className="py-1.5">
                  {e.matches ? (
                    <Badge tone="profit">Matches</Badge>
                  ) : (
                    <Badge
                      tone="loss"
                      title={(e.missing ?? []).map((m) => CHECK_LABEL[m] ?? m).join(', ')}
                    >
                      Misses{' '}
                      {(e.missing ?? []).map((m) => CHECK_LABEL[m] ?? m).join(', ') ||
                        'a condition'}
                    </Badge>
                  )}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function CompetitionTab({ competition: c }: { competition: Competition }) {
  const left = daysLeft(c.endDate)
  const signUpOpen = !c.enrolled && (daysLeft(c.signUpEndDate) ?? 0) > 0
  return (
    <>
      <Panel
        title={c.name}
        actions={
          <>
            <Badge tone={c.enrolled ? 'profit' : 'warn'}>
              {c.enrolled ? 'Enrolled' : 'Not enrolled'}
            </Badge>
            {c.faq && (
              <Button
                variant="ghost"
                size="sm"
                render={<a href={c.faq} target="_blank" rel="noopener noreferrer" />}
              >
                Rules
              </Button>
            )}
            <Button
              size="sm"
              render={
                <a href={BRAIN_COMPETITION_URL(c.id)} target="_blank" rel="noopener noreferrer" />
              }
            >
              <ExternalLinkIcon aria-hidden />
              Open in BRAIN
            </Button>
          </>
        }
      >
        <div className="flex flex-col gap-5">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Metric boxed label="Ends" value={fmt.date(c.endDate)} />
            <Metric boxed label="Days Left" value={left === null ? '—' : fmt.int(left)} />
            <Metric
              boxed
              label="Your Rank"
              value={c.standing?.rank ? `#${fmt.int(c.standing.rank)}` : '—'}
            />
            <Metric boxed label="Your Alphas" value={fmt.int(c.standing?.alphas)} />
          </div>
          {signUpOpen && (
            <p className="text-body text-ink-subtle">
              Sign up on BRAIN by {fmt.date(c.signUpEndDate)} for your Alphas to count. Kepler never
              signs up for you.
            </p>
          )}
        </div>
      </Panel>
      <Panel
        title="Your Entries"
        description={`Submitted Alphas tagged ${c.id}. Only those carrying the tag when the cycle closes count.`}
      >
        <Entries entries={c.entries ?? []} />
      </Panel>
      <PlaybookPanel place={placeOf(c.id)} limit={10} />
    </>
  )
}

function ThemeTabBody({ theme: t }: { theme: ThemeTab }) {
  const left = daysLeft(t.endDate ? `${t.endDate}T23:59:59-05:00` : null)
  const matching = t.entries.filter((e) => e.matches).length
  return (
    <>
      <Panel
        title={t.name}
        description={
          t.requires.length
            ? `Asks for ${t.region} D${t.delay}${t.qualifier ? ` ${t.qualifier}` : ''} and ${t.requires.map((r) => CHECK_LABEL[r] ?? r).join(', ')}.`
            : `Asks for ${t.region} D${t.delay}${t.qualifier ? ` ${t.qualifier}` : ''}.`
        }
        actions={<Badge tone="outline">×{fmt.ratio(t.multiplier, 1)} multiplier</Badge>}
      >
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Metric
            boxed
            label="Runs"
            value={
              t.startDate && t.endDate ? `${fmt.date(t.startDate)} – ${fmt.date(t.endDate)}` : '—'
            }
          />
          <Metric boxed label="Days Left" value={left === null ? '—' : fmt.int(left)} />
          <Metric
            boxed
            label="Matching Alphas"
            value={`${fmt.int(matching)} of ${fmt.int(t.minForAwards)} needed`}
            tone={matching >= t.minForAwards ? 'profit' : 'warn'}
          />
          <Metric boxed label="Tagged in Market" value={fmt.int(t.entries.length)} />
        </div>
      </Panel>
      <Panel
        title="Your Power Pool Alphas in This Market"
        description={`Tagged PowerPoolSelected. Only Alphas submitted after the theme began count; awards need at least ${t.minForAwards} tagged, theme-matching Alphas when it ends.`}
      >
        <Entries entries={t.entries} theme />
      </Panel>
      <PlaybookPanel place={themePlace(t)} limit={10} />
    </>
  )
}

function PastTab({ past }: { past: Competition[] }) {
  return (
    <Panel title="Past Competitions">
      <ul className="flex flex-col gap-2 text-body">
        {past.map((c) => (
          <li key={c.id} className="flex flex-wrap items-baseline justify-between gap-x-4">
            <span>{c.name}</span>
            <span className="text-body-compact text-ink-subtle">
              {fmt.date(c.startDate)} – {fmt.date(c.endDate)}
              {c.enrolled ? ' · enrolled' : ''}
            </span>
          </li>
        ))}
      </ul>
    </Panel>
  )
}

export function CompetitionsScreen() {
  const list = useQuery({ queryKey: ['competitions'], queryFn: competitions.list })
  const challengeList = useQuery({
    queryKey: ['competitions', 'challenges'],
    queryFn: competitions.challenges,
    refetchInterval: 60_000,
  })
  const challenges = challengeList.data?.challenges.filter((c) => !c.ended) ?? []
  const params = useParams({ strict: false }) as { tab?: string }
  const all = list.data?.competitions ?? []
  const live = all.filter((c) => c.ongoing)
  const past = all.filter((c) => !c.ongoing)
  const themes = list.data?.themes ?? []

  const tabs = [
    ...challenges.map((c) => ({ key: c.id, label: c.name })),
    ...live.map((c) => ({ key: c.id.toLowerCase(), label: c.name })),
    ...themes.map((t) => ({ key: themeKey(t), label: t.name })),
    ...(past.length ? [{ key: 'past', label: `Past (${past.length})` }] : []),
  ]
  const tab = params.tab ?? tabs[0]?.key
  const competition = live.find((c) => c.id.toLowerCase() === tab)
  const theme = themes.find((t) => themeKey(t) === tab)
  const challenge = challenges.find((c) => c.id === tab)

  return (
    <Page>
      <PageHeader
        title="Competitions"
        description="Each competition BRAIN is running and each Power Pool theme, with where you stand and what WorldQuant has said about it. Joining and submitting happen on BRAIN."
      />
      {list.isError && <ErrorNotice error={list.error} title="Could not read competitions" />}
      {list.isPending && challengeList.isPending ? (
        <Skeleton label="Reading competitions from BRAIN" />
      ) : tabs.length === 0 ? (
        <Panel>
          <Empty icon={<TrophyIcon />} title="No competition is running right now." />
        </Panel>
      ) : (
        <>
          <TabBar>
            {tabs.map((t) => (
              <TabLink
                key={t.key}
                to="/competitions/$tab"
                params={{ tab: t.key }}
                data-status={t.key === tab ? 'active' : undefined}
              >
                {t.label}
              </TabLink>
            ))}
          </TabBar>
          {challenge ? (
            <ChallengeTabBody challenge={challenge} />
          ) : competition ? (
            <CompetitionTab competition={competition} />
          ) : theme ? (
            <ThemeTabBody theme={theme} />
          ) : tab === 'past' ? (
            <PastTab past={past} />
          ) : (
            <Panel>
              <Empty title={`No competition called “${tab}” is running`}>
                <Link to="/competitions" className={LINK}>
                  Go to the first one
                </Link>
              </Empty>
            </Panel>
          )}
        </>
      )}
    </Page>
  )
}
