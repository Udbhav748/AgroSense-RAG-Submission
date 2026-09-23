import { useRef, useState } from 'react'
import SourceReferences from './SourceReferences'

// Matches every individual [N] bracket the backend was instructed to emit
// (see prompt_builder._INSTRUCTIONS) — each one is checked against the
// actual sources list below before being treated as a real citation, so a
// stray "[1]" the model wrote for some other reason (or one from a source
// index that doesn't exist) just renders as plain text instead of a badge
// that leads nowhere.
const CITATION_RE = /\[(\d+)\]/g

export default function CitedAnswer({ text, sources = [], query }) {
  const [expandedIds, setExpandedIds] = useState(() => new Set())
  const sourceRefs = useRef({})

  const sourceByNumber = new Map(sources.map((source) => [source.number, source]))

  const handleToggle = (chunkId) => {
    setExpandedIds((current) => {
      const next = new Set(current)
      if (next.has(chunkId)) {
        next.delete(chunkId)
      } else {
        next.add(chunkId)
      }
      return next
    })
  }

  // Clicking an inline citation always reveals its source (never
  // collapses it) and scrolls it into view — distinct from clicking a
  // source card's own header, which toggles.
  const openAndScrollTo = (chunkId) => {
    setExpandedIds((current) => new Set(current).add(chunkId))
    requestAnimationFrame(() => {
      sourceRefs.current[chunkId]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    })
  }

  const parts = []
  let lastIndex = 0
  CITATION_RE.lastIndex = 0
  let match = CITATION_RE.exec(text)
  while (match !== null) {
    const number = Number(match[1])
    const source = sourceByNumber.get(number)
    if (source) {
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index))
      }
      parts.push(
        <button
          key={`${match.index}-${number}`}
          type="button"
          onClick={() => openAndScrollTo(source.chunk_id)}
          className="mx-0.5 inline-flex h-4 min-w-[1rem] translate-y-[-1px] items-center justify-center rounded bg-accent-500/15 px-1 align-middle text-[10px] font-semibold text-accent-600 transition-colors hover:bg-accent-500/25 dark:text-accent-400"
        >
          {number}
        </button>
      )
      lastIndex = match.index + match[0].length
    }
    match = CITATION_RE.exec(text)
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  return (
    <>
      <p className="whitespace-pre-wrap leading-relaxed">{parts}</p>
      <SourceReferences
        sources={sources}
        expandedIds={expandedIds}
        onToggle={handleToggle}
        sourceRefs={sourceRefs}
        query={query}
      />
    </>
  )
}
