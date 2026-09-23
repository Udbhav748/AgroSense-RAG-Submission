import { motion } from 'framer-motion'
import { Activity, AlertTriangle, CheckCircle2, RefreshCw, Terminal } from 'lucide-react'
import Button from '../ui/Button'

/**
 * Service Health Warning & Live Indicator for the LeafSense Vision Service (port 8001).
 * Shows live status and provides an actionable troubleshooting guide if offline.
 */
export default function ServiceHealthBanner({
  isOnline = true,
  isChecking = false,
  onRecheck,
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full"
    >
      {isOnline ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/5 px-4 py-2.5 text-xs text-emerald-800 dark:border-emerald-500/20 dark:bg-emerald-500/10 dark:text-emerald-300">
          <div className="flex items-center gap-2.5">
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
            </span>
            <div className="flex items-center gap-1.5 font-medium">
              <CheckCircle2 size={14} className="text-emerald-700 dark:text-emerald-400" />
              <span>LeafSense Vision Engine Active</span>
              <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-800 dark:text-emerald-300">
                Port 8001
              </span>
            </div>
            <span className="hidden text-emerald-700/80 sm:inline dark:text-emerald-400/80">
              · 38 PlantVillage disease classes ready
            </span>
          </div>

          {onRecheck && (
            <button
              type="button"
              onClick={onRecheck}
              disabled={isChecking}
              className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 underline-offset-2 hover:underline disabled:opacity-50 dark:text-emerald-400"
            >
              <RefreshCw size={11} className={isChecking ? 'animate-spin' : ''} />
              Recheck
            </button>
          )}
        </div>
      ) : (
        <div
          data-testid="leafsense-offline-banner"
          className="rounded-panel border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200"
        >
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div className="flex items-start gap-3">
              <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-amber-500/20 text-amber-600 dark:text-amber-400">
                <AlertTriangle size={18} strokeWidth={2} />
              </div>
              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <h4 className="font-semibold text-amber-950 dark:text-amber-100">
                    LeafSense Vision Service Offline (Port 8001)
                  </h4>
                  <span className="rounded border border-amber-500/30 bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800 dark:text-amber-300">
                    Connection Required
                  </span>
                </div>
                <p className="text-xs text-amber-800/90 dark:text-amber-200/90">
                  The deep learning classifier (Hybrid CBAM+ViT+EfficientNet) on port 8001 is unreachable. Leaf photo analysis will fail until the service is running.
                </p>
              </div>
            </div>

            {onRecheck && (
              <Button
                variant="secondary"
                size="sm"
                icon={RefreshCw}
                loading={isChecking}
                onClick={onRecheck}
                className="shrink-0 self-start border-amber-500/30 text-amber-900 hover:bg-amber-500/20 dark:text-amber-100"
              >
                Recheck Port 8001
              </Button>
            )}
          </div>

          <div className="mt-3 rounded-lg border border-amber-500/20 bg-slate-950/80 p-3 font-mono text-xs text-amber-200">
            <div className="mb-1.5 flex items-center gap-1.5 text-[11px] text-amber-400/80">
              <Terminal size={12} />
              <span>Actionable Guide: Start LeafSense locally</span>
            </div>
            <p className="text-slate-300">
              <span className="text-slate-500"># Start LeafSense on port 8001:</span>
              <br />
              <span className="text-emerald-400">cd</span> LeafSense/backend
              <br />
              <span className="text-amber-300">uvicorn main:app --host localhost --port 8001</span>
              <br />
              <span className="text-slate-500"># Or start all full-stack services together:</span>
              <br />
              <span className="text-accent-400">./start-local.ps1</span>
            </p>
          </div>
        </div>
      )}
    </motion.div>
  )
}
