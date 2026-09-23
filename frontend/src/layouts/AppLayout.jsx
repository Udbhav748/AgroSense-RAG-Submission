import { Suspense, useEffect, useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Sidebar from '../components/layout/Sidebar'
import Navbar from '../components/layout/Navbar'
import CommandPalette from '../components/command/CommandPalette'
import PageFallback from '../components/ui/PageFallback'

export default function AppLayout() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)
  const location = useLocation()

  // Global Cmd/Ctrl+K — mounted here (not per-page) so it works from
  // anywhere in the authenticated app, not just whichever page happens to
  // render it. metaKey covers Mac; ctrlKey covers Windows/Linux.
  useEffect(() => {
    const handleKeyDown = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setCommandPaletteOpen((open) => !open)
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  return (
    <div className="min-h-screen bg-surface-light dark:bg-surface-dark print:!bg-white">
      <div className="mx-auto flex max-w-[1400px] gap-4 p-4">
        <Sidebar isOpen={mobileNavOpen} onClose={() => setMobileNavOpen(false)} />

        <div className="flex min-w-0 flex-1 flex-col gap-4">
          <Navbar onMenuClick={() => setMobileNavOpen(true)} onSearchClick={() => setCommandPaletteOpen(true)} />
          <main className="min-w-0 flex-1 pb-4 print:pb-0">
            {/* Only the routed page content animates/remounts on navigation —
                Sidebar/Navbar/CommandPalette above stay mounted, so their
                state survives and no page ever re-fetches data it already
                had just because a sibling route changed. */}
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              >
                <Suspense fallback={<PageFallback />}>
                  <Outlet />
                </Suspense>
              </motion.div>
            </AnimatePresence>
          </main>
        </div>
      </div>

      <CommandPalette open={commandPaletteOpen} onClose={() => setCommandPaletteOpen(false)} />
    </div>
  )
}
