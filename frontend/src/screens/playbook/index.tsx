/**
 * The Playbook: every insight WorldQuant has shared (webinars, FAQ, competition rules) and
 * every lesson this account has measured, found by typing a question. Search is lexical (BM25)
 * on this machine: no model, no cost.
 */

import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { BookOpenIcon, SearchIcon } from 'lucide-react'
import { useDeferredValue, useState } from 'react'
import { fmt } from '@/lib/format'
import {
  Badge,
  Chips,
  Empty,
  ErrorNotice,
  Input,
  Page,
  PageHeader,
  Panel,
  Skeleton,
} from '@/ui/kit'
import { PLAYBOOK_KEY, playbookApi } from './api'
import { InsightBody } from './insight'

const EXAMPLES = [
  'full time role osmosis',
  'raise margin',
  'unused fields multiplier',
  'AQP description',
  'power pool tag',
]

export function PlaybookScreen() {
  const [text, setText] = useState('')
  const [topics, setTopics] = useState<string[]>([])
  const q = useDeferredValue(text.trim())
  const topic = topics[0]
  const query = useQuery({
    queryKey: [...PLAYBOOK_KEY, 'search', q, topic ?? ''],
    queryFn: () => playbookApi.search({ q, ...(topic ? { topic } : {}), limit: 100 }),
    placeholderData: keepPreviousData,
  })
  const data = query.data

  return (
    <Page>
      <PageHeader
        title="Playbook"
        description="What WorldQuant has told consultants in webinars, FAQs and competition rules, and what this account has measured. Ask in your own words; Kepler's chat and screens read the same insights."
      />
      <Panel bodyClassName="flex flex-col gap-3">
        <label className="relative flex items-center">
          <SearchIcon className="pointer-events-none absolute left-2.5 size-4 text-ink-subtle" />
          <Input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Ask anything, e.g. how do I get shortlisted for a full-time role?"
            className="pl-8"
            aria-label="Search the playbook"
          />
        </label>
        <div className="flex flex-wrap items-center gap-1.5 text-body-compact text-ink-subtle">
          Try:
          {EXAMPLES.map((e) => (
            <button
              key={e}
              type="button"
              onClick={() => setText(e)}
              className="rounded-xs border border-hairline px-1.5 py-0.5 hover:text-ink"
            >
              {e}
            </button>
          ))}
        </div>
        {data && data.topics.length > 0 && (
          <Chips
            label="Topic"
            items={data.topics.map((t) => ({ value: t, label: t.replace('_', ' ') }))}
            value={topics}
            onChange={(next) => setTopics(next.filter((t) => !topics.includes(t)).slice(0, 1))}
          />
        )}
      </Panel>

      {query.isError ? (
        <ErrorNotice error={query.error} />
      ) : !data ? (
        <Skeleton className="h-96" label="Loading the playbook" />
      ) : data.total === 0 ? (
        <Panel>
          <Empty title="The playbook is empty" icon={<BookOpenIcon />}>
            Insights are kept in {data.path} on this machine.
          </Empty>
        </Panel>
      ) : data.insights.length === 0 ? (
        <Panel>
          <Empty title="Nothing matches" icon={<SearchIcon />}>
            No insight shares those words. Try fewer or plainer ones.
          </Empty>
        </Panel>
      ) : (
        <>
          <p className="px-1 text-body-compact text-ink-subtle">
            {q
              ? `${data.insights.length} of ${data.total} insights match, most relevant first.`
              : `${data.total} insights, newest first. Kept in ${data.path}, never in the repository.`}
          </p>
          <div className="grid gap-4 xl:grid-cols-2">
            {data.insights.map((insight) => (
              <Panel
                key={insight.id}
                title={insight.title}
                actions={
                  insight.score != null ? (
                    <Badge tone="outline" title="BM25 relevance to your question">
                      {fmt.ratio(insight.score, 1)}
                    </Badge>
                  ) : undefined
                }
              >
                <InsightBody insight={insight} />
              </Panel>
            ))}
          </div>
        </>
      )}
    </Page>
  )
}
