import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { MessageSquareText, Trash2 } from 'lucide-react'
import EmptyState from '../components/ui/EmptyState'
import Button from '../components/ui/Button'
import Modal from '../components/ui/Modal'
import Skeleton from '../components/ui/Skeleton'
import { deleteChatSession, listSessions } from '../services/chatService'
import useToast from '../hooks/useToast'
import getErrorMessage from '../utils/errorMessage'

function formatDate(iso) {
  if (!iso) return 'Unknown date'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return 'Unknown date'
  return date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export default function History() {
  const [sessions, setSessions] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [pendingDelete, setPendingDelete] = useState(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const navigate = useNavigate()
  const { showToast } = useToast()

  useEffect(() => {
    listSessions()
      .then((data) => setSessions(data.sessions))
      .catch((error) => showToast(getErrorMessage(error, 'Failed to load chat history.'), 'error'))
      .finally(() => setIsLoading(false))
    // showToast's identity is stable (useToast), safe to omit — this
    // should only run once, on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const openSession = (session) => {
    navigate('/chat', { state: { sessionId: session.session_id } })
  }

  const confirmDelete = async () => {
    if (!pendingDelete) return
    setIsDeleting(true)
    try {
      await deleteChatSession(pendingDelete.session_id)
      setSessions((current) => current.filter((s) => s.session_id !== pendingDelete.session_id))
      setPendingDelete(null)
    } catch (error) {
      showToast(getErrorMessage(error, 'Failed to delete conversation. Please try again.'), 'error')
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className="space-y-5 py-4">
      <div>
        <h2 className="font-display text-xl font-bold">History</h2>
        <p className="text-sm text-slate-500 dark:text-ink-muted">
          {isLoading
            ? 'Loading…'
            : `${sessions.length} past conversation${sessions.length === 1 ? '' : 's'}`}
        </p>
      </div>

      {isLoading ? (
        <div className="panel divide-y divide-border-light overflow-hidden dark:divide-border">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="flex items-center gap-3 px-5 py-4">
              <Skeleton className="h-10 w-10 shrink-0 rounded-lg" />
              <div className="min-w-0 flex-1 space-y-2">
                <Skeleton className="h-3.5 w-2/5" />
                <Skeleton className="h-3 w-1/4" />
              </div>
            </div>
          ))}
        </div>
      ) : sessions.length === 0 ? (
        <EmptyState
          icon={MessageSquareText}
          title="No past conversations yet"
          description="Conversations you have in Chat will show up here so you can come back to them later."
        />
      ) : (
        <div className="panel divide-y divide-border-light overflow-hidden dark:divide-border">
          {sessions.map((session, index) => (
            <motion.div
              key={session.session_id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.25, delay: index * 0.03 }}
              className="flex items-center gap-3 px-5 py-4 transition-colors hover:bg-slate-900/[0.02] dark:hover:bg-white/[0.02]"
            >
              <button
                type="button"
                onClick={() => openSession(session)}
                className="flex min-w-0 flex-1 items-center gap-3 text-left"
              >
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border-light bg-slate-900/5 text-slate-500 dark:border-border dark:bg-white/[0.03] dark:text-ink-secondary">
                  <MessageSquareText size={18} strokeWidth={1.5} />
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-slate-800 dark:text-ink-primary">
                    {session.title || 'Untitled conversation'}
                  </p>
                  <p className="text-xs text-slate-400 dark:text-ink-muted">
                    {formatDate(session.last_accessed_at)}
                  </p>
                </div>
              </button>
              <button
                type="button"
                onClick={() => setPendingDelete(session)}
                aria-label="Delete conversation"
                className="shrink-0 rounded-lg p-2 text-slate-400 transition-colors hover:bg-danger/10 hover:text-danger dark:text-ink-muted"
              >
                <Trash2 size={15} />
              </button>
            </motion.div>
          ))}
        </div>
      )}

      <Modal
        open={!!pendingDelete}
        onClose={() => setPendingDelete(null)}
        title="Delete conversation"
        footer={
          <>
            <Button variant="secondary" onClick={() => setPendingDelete(null)} disabled={isDeleting}>
              Cancel
            </Button>
            <Button variant="danger" onClick={confirmDelete} loading={isDeleting}>
              Delete
            </Button>
          </>
        }
      >
        Delete{' '}
        <span className="font-medium text-slate-800 dark:text-ink-primary">
          {pendingDelete?.title || 'this conversation'}
        </span>
        ? This can't be undone.
      </Modal>
    </div>
  )
}
