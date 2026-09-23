import { motion } from 'framer-motion'

export default function ProgressBar({ value = 0, indeterminate = false, className = '' }) {
  const clamped = Math.max(0, Math.min(100, value))

  return (
    <div
      className={`h-2 w-full overflow-hidden rounded-full bg-slate-900/5 dark:bg-white/5 ${className}`}
      role="progressbar"
      aria-valuenow={indeterminate ? undefined : clamped}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      {indeterminate ? (
        <div
          className="h-full w-full rounded-full bg-gradient-to-r from-transparent via-accent-400 to-transparent bg-[length:200%_100%] animate-shimmer"
        />
      ) : (
        <motion.div
          className="h-full rounded-full bg-accent-500"
          initial={{ width: 0 }}
          animate={{ width: `${clamped}%` }}
          transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        />
      )}
    </div>
  )
}
