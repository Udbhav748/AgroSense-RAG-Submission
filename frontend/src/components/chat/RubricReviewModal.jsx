import { useState } from 'react'
import { Star } from 'lucide-react'
import Modal from '../ui/Modal'
import Button from '../ui/Button'
import { sendFeedback } from '../../services/feedbackService'
import useToast from '../../hooks/useToast'
import getErrorMessage from '../../utils/errorMessage'

// Mirrors RubricScores in app/models/schemas.py — same seven criteria,
// same 1-5 scale. Keep the two in sync if either changes.
const CRITERIA = [
  { key: 'correctness', label: 'Correctness', hint: 'Is the answer factually and logically right?' },
  { key: 'helpfulness', label: 'Helpfulness', hint: 'Does it help you complete your task?' },
  { key: 'completeness', label: 'Completeness', hint: 'Does it cover every required part?' },
  { key: 'safety', label: 'Safety', hint: 'Does it avoid harmful or risky behavior?' },
  { key: 'tone', label: 'Tone', hint: 'Is the communication appropriate for the context?' },
  { key: 'groundedness', label: 'Groundedness', hint: 'Are its factual claims supported by evidence?' },
  { key: 'citation_quality', label: 'Citation quality', hint: 'Do citations point to the right sources?' },
]

function StarRating({ value, onChange }) {
  return (
    <div className="flex shrink-0 items-center gap-0.5" role="radiogroup">
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          type="button"
          role="radio"
          aria-checked={value === n}
          aria-label={`${n} star${n > 1 ? 's' : ''}`}
          onClick={() => onChange(n)}
          className="rounded p-0.5 transition-transform hover:scale-110"
        >
          <Star
            size={18}
            strokeWidth={1.75}
            className={value >= n ? 'fill-accent-500 text-accent-500' : 'text-slate-300 dark:text-ink-muted'}
          />
        </button>
      ))}
    </div>
  )
}

export default function RubricReviewModal({ open, onClose, messageId, rating, onSubmitted }) {
  const [scores, setScores] = useState({})
  const [submitting, setSubmitting] = useState(false)
  const { showToast } = useToast()

  const allRated = CRITERIA.every((criterion) => scores[criterion.key])

  const setScore = (key, value) => setScores((current) => ({ ...current, [key]: value }))

  const handleClose = () => {
    if (submitting) return
    onClose()
  }

  const handleSubmit = async () => {
    if (!allRated || submitting) return
    setSubmitting(true)
    try {
      await sendFeedback(messageId, rating, null, scores)
      showToast('Detailed review submitted — thank you.', 'success')
      setScores({})
      onSubmitted?.()
      onClose()
    } catch (error) {
      showToast(getErrorMessage(error, 'Could not submit your review.'), 'error')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Detailed review"
      footer={
        <>
          <Button variant="secondary" size="sm" onClick={handleClose} disabled={submitting}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" onClick={handleSubmit} loading={submitting} disabled={!allRated}>
            Submit review
          </Button>
        </>
      }
    >
      <p className="mb-4 text-xs text-slate-500 dark:text-ink-muted">
        Rate each criterion from 1 (poor) to 5 (excellent).
      </p>
      <div className="space-y-3.5">
        {CRITERIA.map((criterion) => (
          <div key={criterion.key} className="flex items-center justify-between gap-4">
            <div className="min-w-0">
              <p className="text-sm font-medium text-slate-700 dark:text-ink-secondary">{criterion.label}</p>
              <p className="truncate text-xs text-slate-400 dark:text-ink-muted">{criterion.hint}</p>
            </div>
            <StarRating value={scores[criterion.key] ?? 0} onChange={(value) => setScore(criterion.key, value)} />
          </div>
        ))}
      </div>
    </Modal>
  )
}
