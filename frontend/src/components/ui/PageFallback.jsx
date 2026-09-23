import Skeleton from './Skeleton'

// Used both as the pre-auth full-page fallback (ProtectedRoute, no app
// chrome around it yet) and as the per-route Suspense fallback inside
// AppLayout (sidebar/navbar already visible) — a generic content-shaped
// skeleton reads reasonably in either context, unlike a spinner floating
// alone in whichever of those two layouts it happens to land in.
export default function PageFallback() {
  return (
    <div className="mx-auto max-w-2xl space-y-5 py-4">
      <div className="space-y-2">
        <Skeleton className="h-6 w-40" />
        <Skeleton className="h-3.5 w-64" />
      </div>
      <Skeleton className="h-32 w-full rounded-panel" />
      <Skeleton className="h-32 w-full rounded-panel" />
    </div>
  )
}
