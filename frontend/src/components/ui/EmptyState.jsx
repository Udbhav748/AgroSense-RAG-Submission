import { motion } from 'framer-motion'

export default function EmptyState({ icon: Icon, title, description, action, className = '' }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className={`flex flex-col items-center justify-center rounded-panel border border-dashed border-border-light py-16 text-center dark:border-border ${className}`}
    >
      {Icon && (
        <span className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-border-light bg-slate-900/5 text-slate-400 dark:border-border dark:bg-white/[0.03] dark:text-ink-muted">
          <Icon size={22} strokeWidth={1.5} />
        </span>
      )}
      {title && <h3 className="text-base font-semibold text-slate-800 dark:text-ink-primary">{title}</h3>}
      {description && (
        <p className="mt-1.5 max-w-sm text-sm text-slate-500 dark:text-ink-muted">{description}</p>
      )}
      {action && <div className="mt-5">{action}</div>}
    </motion.div>
  )
}
