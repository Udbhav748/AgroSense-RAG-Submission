const STOPWORDS = new Set([
  'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'does', 'do',
  'in', 'on', 'of', 'to', 'and', 'or', 'for', 'with', 'about', 'can', 'you', 'your',
  'this', 'that', 'it', 'its', 'their', 'from', 'into',
])

// Highlight the meaningful (non-stopword) terms of a user's query wherever
// they appear in a citation excerpt. Returns an array of strings/<mark>
// elements; callers render it directly. Pure — no state, no effects.
export function highlightTerms(excerpt, query) {
  if (!query) return [excerpt]
  const terms = [
    ...new Set(
      query
        .toLowerCase()
        .split(/\W+/)
        .filter((word) => word.length > 2 && !STOPWORDS.has(word))
    ),
  ]
  if (terms.length === 0) return [excerpt]
  const pattern = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
  return excerpt.split(pattern).map((part, index) =>
    terms.some((t) => part.toLowerCase() === t) ? (
      <mark key={index} className="rounded bg-accent-500/20 px-0.5 text-inherit">
        {part}
      </mark>
    ) : (
      part
    )
  )
}
