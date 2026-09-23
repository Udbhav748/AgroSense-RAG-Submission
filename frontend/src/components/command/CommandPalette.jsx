import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Camera,
  Cpu,
  FileText,
  History,
  MessageSquarePlus,
  MessageSquareText,
  Moon,
  Search,
  Settings,
  Sun,
  UploadCloud,
} from 'lucide-react'
import useTheme from '../../hooks/useTheme'
import { listSessions } from '../../services/chatService'

// Mirrors components/ui/Modal.jsx's own focus-trap selector — kept as a
// separate copy rather than imported since Modal doesn't export it, and
// this component doesn't render through Modal (it needs its own list/input
// layout Modal's title+footer chrome doesn't fit).
const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'

/**
 * Global Cmd/Ctrl+K palette — jump to any page, toggle theme, or resume a
 * past conversation by title, all from one place. Opened from AppLayout's
 * global keydown listener and from the Navbar's search trigger (which
 * used to be a permanently-disabled decorative input doing nothing).
 *
 * Past conversations are fetched fresh every time the palette opens
 * rather than cached — this is a lightweight metadata list (GET
 * /chat/sessions), not worth the staleness risk of holding onto it
 * between opens.
 */
export default function CommandPalette({ open, onClose }) {
  const navigate = useNavigate()
  const { theme, setTheme } = useTheme()
  const [query, setQuery] = useState('')
  const [sessions, setSessions] = useState([])
  const [activeIndex, setActiveIndex] = useState(0)
  const inputRef = useRef(null)
  const dialogRef = useRef(null)
  const itemRefs = useRef([])

  const staticItems = useMemo(
    () => [
      { id: 'nav-diagnose', label: 'Leaf Diagnosis (Plant Vision)', icon: Camera, action: () => navigate('/diagnose') },
      { id: 'nav-chat', label: 'New Chat', icon: MessageSquarePlus, action: () => navigate('/chat') },
      { id: 'nav-documents', label: 'Documents', icon: FileText, action: () => navigate('/documents') },
      { id: 'nav-upload', label: 'Upload Document', icon: UploadCloud, action: () => navigate('/upload') },
      { id: 'nav-history', label: 'History', icon: History, action: () => navigate('/history') },
      { id: 'nav-settings', label: 'Settings', icon: Settings, action: () => navigate('/settings') },
      { id: 'nav-architecture', label: 'System Architecture & AI Review', icon: Cpu, action: () => navigate('/architecture') },
      {
        id: 'toggle-theme',
        label: theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme',
        icon: theme === 'dark' ? Sun : Moon,
        action: () => setTheme(theme === 'dark' ? 'light' : 'dark'),
      },
    ],
    [navigate, theme, setTheme]
  )

  useEffect(() => {
    if (!open) return
    setQuery('')
    setActiveIndex(0)
    listSessions()
      .then((data) => setSessions(data.sessions ?? []))
      .catch(() => setSessions([]))
    // Portal content isn't mounted until the same tick this effect runs
    // in, so the input isn't focusable until the next frame.
    requestAnimationFrame(() => inputRef.current?.focus())
  }, [open])

  // Focus trap + background-scroll lock, same pattern as ui/Modal.jsx —
  // this palette is a global, frequently-used surface (Cmd/Ctrl+K from
  // anywhere), so Tab must not be able to walk focus into the page behind
  // the backdrop, and the page underneath must not scroll while it's open.
  useEffect(() => {
    if (!open) return undefined

    const focusables = () => Array.from(dialogRef.current?.querySelectorAll(FOCUSABLE_SELECTOR) ?? [])

    const handleTrapKeyDown = (event) => {
      if (event.key === 'Escape') {
        onClose?.()
        return
      }
      if (event.key !== 'Tab') return

      const items = focusables()
      if (items.length === 0) return
      const first = items[0]
      const last = items[items.length - 1]

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', handleTrapKeyDown)
    document.body.style.overflow = 'hidden'

    return () => {
      document.removeEventListener('keydown', handleTrapKeyDown)
      document.body.style.overflow = ''
    }
  }, [open, onClose])

  // Keep the keyboard-active item visible as arrow keys move it past the
  // edge of the scrollable list — mirrors CitedAnswer.jsx's
  // openAndScrollTo pattern for the same "active item may be off-screen"
  // problem.
  useEffect(() => {
    itemRefs.current[activeIndex]?.scrollIntoView({ block: 'nearest' })
  }, [activeIndex])

  const sessionItems = useMemo(
    () =>
      sessions.map((session) => ({
        id: `session-${session.session_id}`,
        label: session.title || 'Untitled conversation',
        icon: MessageSquareText,
        action: () => navigate('/chat', { state: { sessionId: session.session_id } }),
      })),
    [sessions, navigate]
  )

  const allItems = useMemo(() => [...staticItems, ...sessionItems], [staticItems, sessionItems])

  const filtered = useMemo(() => {
    const trimmed = query.trim().toLowerCase()
    if (!trimmed) return allItems
    return allItems.filter((item) => item.label.toLowerCase().includes(trimmed))
  }, [allItems, query])

  useEffect(() => {
    setActiveIndex(0)
  }, [query])

  const runItem = (item) => {
    item.action()
    onClose()
  }

  const handleKeyDown = (event) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setActiveIndex((current) => Math.min(current + 1, filtered.length - 1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setActiveIndex((current) => Math.max(current - 1, 0))
    } else if (event.key === 'Enter') {
      event.preventDefault()
      const item = filtered[activeIndex]
      if (item) runItem(item)
    } else if (event.key === 'Escape') {
      onClose()
    }
  }

  return createPortal(
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-[95] flex items-start justify-center px-4 pt-[12vh]">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="absolute inset-0 bg-slate-950/50 backdrop-blur-sm"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.97, y: -8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: -8, transition: { duration: 0.12 } }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            ref={dialogRef}
            className="glass-panel relative z-10 w-full max-w-lg overflow-hidden shadow-soft-lg"
            role="dialog"
            aria-modal="true"
            aria-label="Command palette"
          >
            <div className="flex items-center gap-2.5 border-b border-border-light px-4 py-3 dark:border-border">
              <Search size={16} className="shrink-0 text-slate-400 dark:text-ink-muted" />
              <input
                ref={inputRef}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Jump to a page, search conversations..."
                className="w-full bg-transparent text-sm text-slate-800 placeholder-slate-400 outline-none dark:text-ink-primary dark:placeholder-ink-muted"
              />
              <kbd className="shrink-0 rounded border border-border-light px-1.5 py-0.5 text-[10px] text-slate-400 dark:border-border dark:text-ink-muted">
                Esc
              </kbd>
            </div>
            <div className="max-h-80 overflow-y-auto p-2">
              {filtered.length === 0 ? (
                <p className="px-3 py-6 text-center text-sm text-slate-400 dark:text-ink-muted">No matches.</p>
              ) : (
                filtered.map((item, index) => (
                  <button
                    key={item.id}
                    ref={(el) => (itemRefs.current[index] = el)}
                    type="button"
                    onClick={() => runItem(item)}
                    onMouseEnter={() => setActiveIndex(index)}
                    className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${
                      index === activeIndex
                        ? 'bg-accent-500/10 text-accent-600 dark:text-accent-500'
                        : 'text-slate-600 dark:text-ink-secondary'
                    }`}
                  >
                    <item.icon size={15} strokeWidth={1.75} className="shrink-0" />
                    <span className="truncate">{item.label}</span>
                  </button>
                ))
              )}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>,
    document.body
  )
}
