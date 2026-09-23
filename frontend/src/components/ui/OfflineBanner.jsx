import { useEffect, useState } from 'react'
import { Calculator, CheckCircle2, WifiOff } from 'lucide-react'
import { Link } from 'react-router-dom'

/**
 * Offline status indicator banner for rural farm fields and low-connectivity environments.
 * Listens to window online/offline events and displays offline availability of local
 * dosage calculations and cached treatment guides.
 */
export default function OfflineBanner() {
  const [isOffline, setIsOffline] = useState(() => {
    if (typeof navigator !== 'undefined' && typeof navigator.onLine === 'boolean') {
      return !navigator.onLine
    }
    return false
  })
  const [showReconnected, setShowReconnected] = useState(false)

  useEffect(() => {
    const handleOnline = () => {
      setIsOffline(false)
      setShowReconnected(true)
    }

    const handleOffline = () => {
      setIsOffline(true)
      setShowReconnected(false)
    }

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  // Auto-dismiss reconnected notification after 3.5s
  useEffect(() => {
    if (!showReconnected) return
    const timer = setTimeout(() => {
      setShowReconnected(false)
    }, 3500)
    return () => clearTimeout(timer)
  }, [showReconnected])

  if (!isOffline && !showReconnected) {
    return null
  }

  if (isOffline) {
    return (
      <div
        key="offline-banner"
        data-testid="offline-banner"
        role="status"
        aria-live="polite"
        className="sticky top-0 z-50 w-full border-b border-amber-500/30 bg-amber-500/15 px-4 py-2.5 backdrop-blur-md transition-all duration-200 dark:border-amber-400/30 dark:bg-amber-950/80"
      >
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2.5">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-amber-500/20 text-amber-700 dark:text-amber-300">
              <WifiOff size={14} className="stroke-[2.5]" />
            </span>
            <div className="font-medium text-amber-950 dark:text-amber-100">
              <span>Field Offline Mode Active — Local Dosage Calculator and Cached Treatment Guides Available</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Link
              to="/diagnose"
              className="inline-flex items-center gap-1.5 rounded-lg border border-amber-500/30 bg-amber-500/20 px-2.5 py-1 text-[11px] font-semibold text-amber-900 transition-colors hover:bg-amber-500/30 dark:border-amber-400/40 dark:bg-amber-400/10 dark:text-amber-200 dark:hover:bg-amber-400/20"
            >
              <Calculator size={12} />
              <span>Open Field Calculator</span>
            </Link>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      key="reconnected-banner"
      data-testid="reconnected-banner"
      role="status"
      aria-live="polite"
      className="sticky top-0 z-50 w-full border-b border-emerald-500/30 bg-emerald-500/15 px-4 py-2 backdrop-blur-md transition-all duration-200 dark:border-emerald-400/30 dark:bg-emerald-950/80"
    >
      <div className="mx-auto flex max-w-[1400px] items-center justify-center gap-2 text-xs font-medium text-emerald-950 dark:text-emerald-100">
        <CheckCircle2 size={14} className="text-emerald-600 dark:text-emerald-400" />
        <span>Connection Restored — Cloud RAG & LeafSense Sync Active</span>
      </div>
    </div>
  )
}
