/**
 * Content-shaped loading placeholder. Uses the `shimmer` keyframe already
 * defined in tailwind.config.js (previously only wired to ProgressBar's
 * indeterminate state) — a moving gradient sweep across the block, which
 * reads as "real content is arriving" far more convincingly than a
 * generic spinner sitting in empty space.
 *
 * Sizing/shape is entirely up to the caller via className (h-4 w-32,
 * h-10 w-10 rounded-full, etc.) — this component only owns the shimmer
 * treatment itself.
 */
export default function Skeleton({ className = '' }) {
  return (
    <div
      className={`animate-shimmer rounded-md bg-gradient-to-r from-slate-900/5 via-slate-900/10 to-slate-900/5 bg-[length:200%_100%] dark:from-white/[0.04] dark:via-white/[0.09] dark:to-white/[0.04] ${className}`}
      aria-hidden="true"
    />
  )
}
