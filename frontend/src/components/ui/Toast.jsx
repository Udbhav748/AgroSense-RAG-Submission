import { motion } from 'framer-motion'
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react'

const ICONS = {
  success: CheckCircle2,
  error: AlertCircle,
  info: Info,
}

const ACCENTS = {
  success: 'text-success bg-success/10',
  error: 'text-danger bg-danger/10',
  info: 'text-accent-400 bg-accent-400/10',
}

export default function Toast({ message, type = 'info', onDismiss }) {
  const Icon = ICONS[type] ?? Info

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.15 } }}
      transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
      className="glass-panel flex w-80 items-start gap-3 p-4 pr-3 shadow-soft-lg"
    >
      <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${ACCENTS[type] ?? ACCENTS.info}`}>
        <Icon size={16} strokeWidth={2.25} />
      </span>
      <p className="mt-1 flex-1 text-sm leading-snug text-slate-700 dark:text-ink-secondary">{message}</p>
      <button
        type="button"
        onClick={onDismiss}
        className="rounded-lg p-1 text-slate-400 transition-colors hover:bg-slate-900/5 hover:text-slate-600 dark:text-ink-muted dark:hover:bg-white/[0.05] dark:hover:text-ink-primary"
        aria-label="Dismiss notification"
      >
        <X size={14} />
      </button>
    </motion.div>
  )
}
