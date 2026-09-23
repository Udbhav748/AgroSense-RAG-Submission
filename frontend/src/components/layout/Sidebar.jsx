import { useEffect, useState } from 'react'
import { Link, NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Camera,
  Cpu,
  FileText,
  History,
  LogOut,
  MessageSquarePlus,
  Settings,
  Shield,
  UploadCloud,
  X,
} from 'lucide-react'
import ThemeToggle from '../ui/ThemeToggle'
import Logo from '../ui/Logo'
import useAuth from '../../hooks/useAuth'
import { listApprovals } from '../../services/adminService'

const NAV_SECTIONS = [
  {
    title: 'Plant Pathology & Vision',
    items: [
      {
        to: '/diagnose',
        label: 'Leaf Diagnosis',
        icon: Camera,
        badge: 'Vision AI',
      },
    ],
  },
  {
    title: 'Document Intelligence',
    items: [
      { to: '/chat', label: 'New Chat', icon: MessageSquarePlus },
      { to: '/documents', label: 'Documents', icon: FileText },
      { to: '/upload', label: 'Upload Document', icon: UploadCloud },
      { to: '/history', label: 'History', icon: History },
    ],
  },
]

const navItemClass = ({ isActive }) =>
  `group flex items-center gap-3 rounded-lg border-l-2 py-2 pl-[10px] pr-3 text-sm font-medium transition-all duration-200 ${
    isActive
      ? 'border-accent-500 bg-slate-900/5 text-slate-900 dark:bg-white/[0.04] dark:text-ink-primary'
      : 'border-transparent text-slate-500 hover:bg-slate-900/5 hover:text-slate-800 dark:text-ink-muted dark:hover:bg-white/[0.03] dark:hover:text-ink-secondary'
  }`

function SidebarContent({ onNavigate }) {
  const { user, logout } = useAuth()
  const isAdmin = user?.role === 'admin'
  const [pendingApprovals, setPendingApprovals] = useState(0)

  // Pending-approval count is what makes the Admin nav entry worth
  // noticing at a glance, not just another link — without it an admin has
  // no signal that anything is waiting on them short of visiting the page.
  useEffect(() => {
    if (!isAdmin) return undefined
    let active = true
    listApprovals({ status: 'pending' })
      .then((data) => {
        if (active) setPendingApprovals(data.approvals?.length ?? 0)
      })
      .catch(() => {
        // Non-critical — the badge just stays at 0 if this fails.
      })
    return () => {
      active = false
    }
  }, [isAdmin])

  return (
    <div className="flex h-full flex-col">
      <Link
        to="/"
        onClick={onNavigate}
        className="flex items-center gap-2.5 rounded-lg px-5 pb-5 pt-6 transition-opacity hover:opacity-80"
      >
        <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-border-light bg-slate-900/5 text-accent-600 dark:border-border dark:bg-white/[0.03] dark:text-accent-500">
          <Logo size={17} />
        </span>
        <div className="leading-tight">
          <p className="font-display text-sm font-semibold text-slate-900 dark:text-ink-primary">AgroSense</p>
          <p className="text-[11px] text-slate-400 dark:text-ink-muted">AI Agronomy & DocAI</p>
        </div>
      </Link>

      <nav className="flex-1 space-y-4 px-3">
        {NAV_SECTIONS.map((section) => (
          <div key={section.title} className="space-y-1">
            <p className="px-2.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400 dark:text-ink-muted">
              {section.title}
            </p>
            <div className="space-y-0.5">
              {section.items.map(({ to, label, icon: Icon, badge }) => (
                <NavLink key={to} to={to} onClick={onNavigate} className={navItemClass}>
                  <Icon size={16} strokeWidth={1.75} />
                  <span className="flex-1">{label}</span>
                  {badge && (
                    <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-600 dark:border-emerald-500/30 dark:bg-emerald-500/15 dark:text-emerald-400">
                      {badge}
                    </span>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      <div className="space-y-0.5 border-t border-border-light px-3 pb-5 pt-3 dark:border-border">
        {isAdmin && (
          <NavLink to="/admin" onClick={onNavigate} className={navItemClass}>
            <Shield size={17} strokeWidth={1.75} />
            <span className="flex-1">Admin</span>
            {pendingApprovals > 0 && (
              <span className="flex h-5 min-w-5 shrink-0 items-center justify-center rounded-full bg-accent-500 px-1 text-[10px] font-semibold text-white">
                {pendingApprovals}
              </span>
            )}
          </NavLink>
        )}
        <NavLink to="/settings" onClick={onNavigate} className={navItemClass}>
          <Settings size={17} strokeWidth={1.75} />
          Settings
        </NavLink>
        <NavLink to="/architecture" onClick={onNavigate} className={navItemClass}>
          <Cpu size={17} strokeWidth={1.75} />
          Architecture & AI Review
        </NavLink>
        <div className="flex items-center justify-between rounded-lg py-2 pl-[10px] pr-3 text-sm text-slate-500 dark:text-ink-muted">
          Appearance
          <ThemeToggle />
        </div>
        {user && (
          <button
            type="button"
            onClick={logout}
            className="flex w-full items-center gap-3 rounded-lg py-2.5 pl-[10px] pr-3 text-sm font-medium text-slate-500 transition-colors hover:bg-slate-900/5 hover:text-slate-800 dark:text-ink-muted dark:hover:bg-white/[0.03] dark:hover:text-ink-secondary"
          >
            <LogOut size={17} strokeWidth={1.75} />
            <span className="truncate">{user.email}</span>
          </button>
        )}
      </div>
    </div>
  )
}

export default function Sidebar({ isOpen, onClose }) {
  useEffect(() => {
    if (!isOpen) return undefined
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose?.()
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  return (
    <>
      {/* Desktop: static column */}
      <aside className="glass-panel sticky top-4 hidden h-[calc(100vh-2rem)] w-64 shrink-0 lg:block print:hidden">
        <SidebarContent />
      </aside>

      {/* Mobile: off-canvas drawer */}
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 z-40 bg-slate-950/60 backdrop-blur-sm lg:hidden"
              onClick={onClose}
            />
            <motion.aside
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              className="glass-panel fixed inset-y-4 left-4 z-50 w-64 lg:hidden"
            >
              <button
                type="button"
                onClick={onClose}
                aria-label="Close menu"
                className="absolute right-4 top-5 rounded-lg p-1.5 text-slate-400 hover:bg-slate-900/5 hover:text-slate-700 dark:text-ink-muted dark:hover:bg-white/[0.05] dark:hover:text-ink-primary"
              >
                <X size={16} />
              </button>
              <SidebarContent onNavigate={onClose} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  )
}
