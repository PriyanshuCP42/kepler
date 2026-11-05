/**
 * One Playbook insight, and the panel a screen shows its own insights in. The panel lists only
 * titles and summaries; each opens to its points, figures and what to do.
 */

import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { BookOpenIcon } from 'lucide-react'
import { Badge, Disclosure, LINK, Panel } from '@/ui/kit'
import { type Insight, PLAYBOOK_KEY, playbookApi } from './api'

export function InsightBody({ insight }: { insight: Insight }) {
  return (
    <div className="flex flex-col gap-3 text-body text-ink-muted">
      <p className="text-pretty text-ink">{insight.summary}</p>
      {insight.points.length > 0 && (
        <ul className="flex list-disc flex-col gap-1 pl-5">
          {insight.points.map((p) => (
            <li key={p} className="text-pretty">
              {p}
            </li>
          ))}
        </ul>
      )}
      {insight.figures.length > 0 && (
        <dl className="grid grid-cols-[repeat(auto-fill,minmax(13rem,1fr))] gap-x-4 gap-y-1">
          {insight.figures.map(([label, value]) => (
            <div
              key={label}
              className="flex justify-between gap-3 border-b border-hairline py-1 text-body-compact"
            >
              <dt className="text-ink-subtle">{label}</dt>
              <dd className="num text-ink">{value}</dd>
            </div>
          ))}
        </dl>
      )}
      {insight.actions.length > 0 && (
        <div className="flex flex-col gap-1 rounded-sm border border-hairline bg-surface-2 p-3">
          <span className="text-body-compact font-medium text-ink">What to do</span>
          <ul className="flex list-disc flex-col gap-1 pl-5">
            {insight.actions.map((a) => (
              <li key={a} className="text-pretty">
                {a}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-1.5 text-body-compact text-ink-subtle">
        <span>{insight.source}</span>
        {insight.date && <span>· {insight.date}</span>}
        {insight.topics.map((t) => (
          <Badge key={t} tone="outline">
            {t.replace('_', ' ')}
          </Badge>
        ))}
      </div>
    </div>
  )
}

/** The insights tagged for one screen, newest first. Hidden while the playbook is empty. */
export function PlaybookPanel({ place, limit = 6 }: { place: string; limit?: number }) {
  const query = useQuery({
    queryKey: [...PLAYBOOK_KEY, 'place', place, limit],
    queryFn: () => playbookApi.search({ place, limit }),
    staleTime: 60_000,
  })
  const insights = query.data?.insights ?? []
  if (!insights.length) return null
  return (
    <Panel
      title={
        <span className="inline-flex items-center gap-2">
          <BookOpenIcon className="size-4" />
          From the Playbook
        </span>
      }
      description="WorldQuant's own guidance and what this account has measured, for this screen."
      actions={
        <Link to="/playbook" className={LINK}>
          All {query.data?.total} insights
        </Link>
      }
      bodyClassName="flex flex-col gap-2"
    >
      {insights.map((insight) => (
        <Disclosure key={insight.id} summary={insight.title}>
          <InsightBody insight={insight} />
        </Disclosure>
      ))}
    </Panel>
  )
}
