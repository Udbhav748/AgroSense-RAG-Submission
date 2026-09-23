import { AnimatePresence, motion } from 'framer-motion'
import { CheckCircle2, FileText, X, XCircle } from 'lucide-react'
import ProgressBar from '../ui/ProgressBar'

function formatBytes(bytes) {
  if (!bytes) return '0 KB'
  const kb = bytes / 1024
  if (kb < 1024) return `${kb.toFixed(0)} KB`
  return `${(kb / 1024).toFixed(1)} MB`
}

export default function PDFCard({ file, progress = 0, status = 'idle', errorMessage, onRemove }) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.97 }}
      className="panel flex items-center gap-4 p-4"
    >
      <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-border-light bg-slate-900/5 text-slate-500 dark:border-border dark:bg-white/[0.03] dark:text-ink-secondary">
        <FileText size={19} strokeWidth={1.5} />
      </span>

      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <p className="truncate text-sm font-medium text-slate-800 dark:text-ink-primary">{file.name}</p>
          <AnimatePresence mode="wait">
            {status === 'success' && (
              <motion.span
                key="success"
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-success"
              >
                <CheckCircle2 size={18} />
              </motion.span>
            )}
            {status === 'error' && (
              <motion.span
                key="error"
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-danger"
              >
                <XCircle size={18} />
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        <p className="mt-0.5 text-xs text-slate-400">
          {formatBytes(file.size)}
          {status === 'uploading' && ` · Uploading ${progress}%`}
          {status === 'success' && ' · Processed'}
          {status === 'error' && errorMessage ? ` · ${errorMessage}` : ''}
        </p>

        {status === 'uploading' && (
          <div className="mt-2">
            <ProgressBar value={progress} />
          </div>
        )}
      </div>

      {status !== 'uploading' && (
        <button
          type="button"
          onClick={onRemove}
          aria-label="Remove file"
          className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-900/5 hover:text-slate-600 dark:text-ink-muted dark:hover:bg-white/[0.05] dark:hover:text-ink-primary"
        >
          <X size={15} />
        </button>
      )}
    </motion.div>
  )
}
