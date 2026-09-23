/**
 * Custom brand mark — a folded-corner document with a spark overlapping
 * it, standing for "insight found within a document" (this is a document
 * intelligence product, not a generic assistant). Replaces the stock
 * lucide <Sparkles> icon that was previously doing double duty as both
 * the brand mark and a generic "AI content" indicator elsewhere in the
 * app — those other, non-brand usages (assistant chat bubble avatar,
 * "Diagnosis explained" section) intentionally keep Sparkles; this
 * component is only for actual brand/logo placement (Sidebar, Login,
 * Signup, Home hero).
 *
 * Uses currentColor throughout, same drop-in convention every lucide
 * icon in this codebase already follows — sizing/coloring is entirely up
 * to the containing element's text color and the `size` prop.
 */
export default function Logo({ size = 20, className = '' }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      aria-hidden="true"
    >
      <path
        d="M6 2.75h8.5L19 7.25V19.5a1.75 1.75 0 0 1-1.75 1.75H6A1.75 1.75 0 0 1 4.25 19.5V4.5A1.75 1.75 0 0 1 6 2.75z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path d="M14.5 2.75V6.5a0.75 0.75 0 0 0 0.75 0.75H19" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path
        d="M12.5 10.5l1 2.2 2.2 1-2.2 1-1 2.2-1-2.2-2.2-1 2.2-1z"
        fill="currentColor"
      />
    </svg>
  )
}
