import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Compass, Home } from 'lucide-react'
import Button from '../components/ui/Button'

export default function NotFound() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center py-20 text-center"
    >
      <span className="mb-5 flex h-14 w-14 items-center justify-center rounded-xl border border-border-light bg-slate-900/5 text-slate-400 dark:border-border dark:bg-white/[0.03] dark:text-ink-muted">
        <Compass size={24} strokeWidth={1.5} />
      </span>
      {/* Purely decorative ghost numeral — the message below already says
          this in words, so a screen reader gains nothing from "4 0 4". */}
      <h2 aria-hidden="true" className="font-display text-6xl font-bold text-slate-200 dark:text-white/10">
        404
      </h2>
      <p className="mt-3 text-sm text-slate-500 dark:text-ink-muted">
        This page doesn't exist, or it may have moved.
      </p>
      <Button as={Link} to="/" variant="primary" icon={Home} className="mt-6">
        Back home
      </Button>
    </motion.div>
  )
}
