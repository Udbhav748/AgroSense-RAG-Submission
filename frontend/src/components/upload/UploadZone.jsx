import { useCallback, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { UploadCloud } from 'lucide-react'
import { MAX_UPLOAD_SIZE_MB } from '../../constants'

export default function UploadZone({ onFileSelected, disabled }) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef(null)

  const handleFiles = useCallback(
    (fileList) => {
      const file = fileList?.[0]
      if (file) onFileSelected(file)
    },
    [onFileSelected]
  )

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    if (disabled) return
    handleFiles(event.dataTransfer.files)
  }

  const openBrowser = () => !disabled && inputRef.current?.click()

  return (
    <motion.div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload a PDF: drag and drop, or press Enter to browse files"
      onDragOver={(event) => {
        event.preventDefault()
        if (!disabled) setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={openBrowser}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          openBrowser()
        }
      }}
      whileHover={disabled ? {} : { scale: 1.005 }}
      className={`panel flex cursor-pointer flex-col items-center justify-center gap-4 border-2 border-dashed px-6 py-16 text-center transition-colors duration-200
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/50 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-surface-dark ${
        isDragging
          ? 'border-accent-500/60 bg-accent-500/5'
          : 'border-border-light dark:border-border'
      } ${disabled ? 'cursor-not-allowed opacity-60' : ''}`}
    >
      <motion.span
        animate={{ y: isDragging ? -4 : 0 }}
        transition={{ duration: 0.2 }}
        className="flex h-14 w-14 items-center justify-center rounded-xl border border-border-light bg-slate-900/5 text-slate-500 dark:border-border dark:bg-white/[0.03] dark:text-ink-secondary"
      >
        <UploadCloud size={26} strokeWidth={1.5} />
      </motion.span>

      <div>
        <p className="font-display text-base font-semibold text-slate-800 dark:text-ink-primary">
          {isDragging ? 'Drop your PDF here' : 'Drag & drop a PDF, or click to browse'}
        </p>
        <p className="mt-1 text-sm text-slate-500 dark:text-ink-muted">PDF only, up to {MAX_UPLOAD_SIZE_MB}MB</p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        disabled={disabled}
        onChange={(event) => handleFiles(event.target.files)}
      />
    </motion.div>
  )
}
