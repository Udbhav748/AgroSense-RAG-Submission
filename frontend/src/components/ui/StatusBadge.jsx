import { AlertTriangle, CheckCircle2, Clock, Loader2, XCircle } from 'lucide-react'

const STATUS_MAP = {
  uploaded: { label: 'Uploaded', icon: Clock, className: 'bg-slate-500/10 text-slate-500 dark:text-ink-muted' },
  processing: {
    label: 'Processing',
    icon: Loader2,
    className: 'bg-accent-500/10 text-accent-600 dark:text-accent-400',
    spin: true,
  },
  processed: {
    label: 'Processed',
    icon: CheckCircle2,
    className: 'bg-success/10 text-success',
  },
  warning: {
    label: 'No text found',
    icon: AlertTriangle,
    className: 'bg-warning/10 text-warning',
  },
  failed: { label: 'Failed', icon: XCircle, className: 'bg-danger/10 text-danger' },
}

export default function StatusBadge({ status = 'uploaded', className = '' }) {
  const config = STATUS_MAP[status] ?? STATUS_MAP.uploaded
  const Icon = config.icon

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${config.className} ${className}`}
    >
      <Icon size={12} className={config.spin ? 'animate-spin' : ''} strokeWidth={2.5} />
      {config.label}
    </span>
  )
}
