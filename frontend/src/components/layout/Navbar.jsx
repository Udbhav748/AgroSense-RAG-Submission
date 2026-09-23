import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { LogOut, Menu, Search, Settings, Shield, User } from 'lucide-react'
import useAuth from '../../hooks/useAuth'

const TITLES = {
  '/': 'Home',
  '/chat': 'Chat',
  '/history': 'History',
  '/diagnose': 'Leaf Diagnosis',
  '/upload': 'Upload Document',
  '/documents': 'Documents',
  '/settings': 'Settings',
  '/admin': 'Admin',
}

// ⌘K on macOS, Ctrl+K everywhere else — match the hint to the actual
// modifier the user needs to press (CommandPalette listens for both).
// navigator.platform is deprecated and unreliable on some browsers; prefer
// the modern userAgentData API, falling back to platform for legacy engines.
const navPlatform = navigator.userAgentData?.platform ?? navigator.platform ?? ''
const SHORTCUT_HINT = /Mac|iPhone|iPod|iPad/i.test(navPlatform) ? '⌘K' : 'Ctrl K'

export default function Navbar({ onMenuClick, onSearchClick }) {
  const { pathname } = useLocation()
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const title = TITLES[pathname] ?? 'AgroSense-RAG'

  const [profileOpen, setProfileOpen] = useState(false)
  const profileRef = useRef(null)

  // Close the profile menu on outside click / Escape, same pattern as
  // Chat.jsx's export menu.
  useEffect(() => {
    if (!profileOpen) return undefined
    const handlePointerDown = (event) => {
      if (profileRef.current && !profileRef.current.contains(event.target)) {
        setProfileOpen(false)
      }
    }
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') setProfileOpen(false)
    }
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [profileOpen])

  const handleLogout = () => {
    setProfileOpen(false)
    logout()
    navigate('/login')
  }

  const goToSettings = () => {
    setProfileOpen(false)
    navigate('/settings')
  }

  const goToAdmin = () => {
    setProfileOpen(false)
    navigate('/admin')
  }

  return (
    <header className="glass-panel sticky top-4 z-30 flex items-center gap-4 px-4 py-3 sm:px-6 print:hidden">
      <button
        type="button"
        onClick={onMenuClick}
        aria-label="Open menu"
        className="rounded-lg p-2 text-slate-500 hover:bg-slate-900/5 hover:text-slate-800 dark:text-ink-muted dark:hover:bg-white/[0.05] dark:hover:text-ink-primary lg:hidden"
      >
        <Menu size={18} strokeWidth={1.75} />
      </button>

      <h1 className="font-display text-base font-semibold text-slate-900 dark:text-ink-primary sm:text-lg">
        {title}
      </h1>

      <div className="ml-auto flex items-center gap-3">
        {/* Was a permanently-disabled decorative <input> — now the actual
            trigger for the command palette (Cmd/Ctrl+K), which is where
            typing/searching genuinely happens. A button styled to look
            like a search field, not a real text input, since this never
            needs its own text state. */}
        <button
          type="button"
          onClick={onSearchClick}
          className="relative hidden h-9 w-48 items-center rounded-lg border border-border-light bg-white/50 pl-9 pr-2 text-left text-sm text-slate-400 transition-colors hover:bg-white/80 dark:border-border dark:bg-white/[0.03] dark:text-ink-muted dark:hover:bg-white/[0.06] sm:flex"
        >
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2" />
          <span className="flex-1 truncate">Search...</span>
          <kbd className="shrink-0 rounded border border-border-light px-1.5 py-0.5 text-[10px] dark:border-border">
            {SHORTCUT_HINT}
          </kbd>
        </button>

        <div className="relative" ref={profileRef}>
          <button
            type="button"
            onClick={() => setProfileOpen((open) => !open)}
            aria-label="Profile"
            aria-haspopup="menu"
            aria-expanded={profileOpen}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-border-light bg-slate-100 text-slate-600 transition-colors hover:bg-slate-200 dark:border-border dark:bg-white/[0.04] dark:text-ink-secondary dark:hover:bg-white/[0.07]"
          >
            <User size={16} strokeWidth={1.75} />
          </button>
          <AnimatePresence>
            {profileOpen && (
              <motion.div
                initial={{ opacity: 0, y: 4, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 4, scale: 0.98 }}
                transition={{ duration: 0.12 }}
                role="menu"
                className="panel absolute right-0 top-full z-40 mt-2 w-56 overflow-hidden rounded-xl p-1"
              >
                <div className="border-b border-border-light px-3 py-2.5 dark:border-border">
                  <p className="truncate text-sm font-medium text-slate-800 dark:text-ink-primary">
                    {user?.email ?? 'Signed in'}
                  </p>
                  <p className="truncate text-[11px] text-slate-400 dark:text-ink-muted">
                    {user?.role === 'admin' ? 'Administrator' : 'Member'}
                  </p>
                </div>
                <button
                  type="button"
                  role="menuitem"
                  onClick={goToSettings}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs text-slate-600 transition-colors hover:bg-slate-100 dark:text-ink-secondary dark:hover:bg-white/[0.05]"
                >
                  <Settings size={14} strokeWidth={1.75} />
                  Settings
                </button>
                {user?.role === 'admin' && (
                  <button
                    type="button"
                    role="menuitem"
                    onClick={goToAdmin}
                    className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs text-slate-600 transition-colors hover:bg-slate-100 dark:text-ink-secondary dark:hover:bg-white/[0.05]"
                  >
                    <Shield size={14} strokeWidth={1.75} />
                    Admin
                  </button>
                )}
                <button
                  type="button"
                  role="menuitem"
                  onClick={handleLogout}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs text-danger transition-colors hover:bg-red-50 dark:hover:bg-red-500/10"
                >
                  <LogOut size={14} strokeWidth={1.75} />
                  Log out
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  )
}
