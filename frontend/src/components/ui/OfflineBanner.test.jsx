import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import OfflineBanner from './OfflineBanner'

describe('OfflineBanner - PWA Field Resilience & Connectivity Indicator', () => {
  const originalOnLine = navigator.onLine

  beforeEach(() => {
    vi.useFakeTimers()
    Object.defineProperty(navigator, 'onLine', {
      value: true,
      writable: true,
      configurable: true,
    })
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
    Object.defineProperty(navigator, 'onLine', {
      value: originalOnLine,
      writable: true,
      configurable: true,
    })
  })

  it('does not display the offline banner when browser is online by default', () => {
    render(
      <MemoryRouter>
        <OfflineBanner />
      </MemoryRouter>
    )

    expect(screen.queryByTestId('offline-banner')).not.toBeInTheDocument()
    expect(
      screen.queryByText(/Field Offline Mode Active — Local Dosage Calculator and Cached Treatment Guides Available/i)
    ).not.toBeInTheDocument()
  })

  it('displays the offline banner immediately when initial navigator.onLine is false', () => {
    Object.defineProperty(navigator, 'onLine', {
      value: false,
      writable: true,
      configurable: true,
    })

    render(
      <MemoryRouter>
        <OfflineBanner />
      </MemoryRouter>
    )

    const banner = screen.getByTestId('offline-banner')
    expect(banner).toBeInTheDocument()
    expect(
      screen.getByText('Field Offline Mode Active — Local Dosage Calculator and Cached Treatment Guides Available')
    ).toBeInTheDocument()

    // Verifies link to local dosage calculator and treatment guides
    const calculatorLink = screen.getByRole('link', { name: /Open Field Calculator/i })
    expect(calculatorLink).toBeInTheDocument()
    expect(calculatorLink).toHaveAttribute('href', '/diagnose')
  })

  it('transitions to offline mode and displays field banner when window offline event is dispatched', () => {
    render(
      <MemoryRouter>
        <OfflineBanner />
      </MemoryRouter>
    )

    expect(screen.queryByTestId('offline-banner')).not.toBeInTheDocument()

    // Trigger offline event (e.g. entering low-connectivity farm field)
    act(() => {
      window.dispatchEvent(new Event('offline'))
    })

    expect(screen.getByTestId('offline-banner')).toBeInTheDocument()
    expect(
      screen.getByText('Field Offline Mode Active — Local Dosage Calculator and Cached Treatment Guides Available')
    ).toBeInTheDocument()
  })

  it('transitions from offline to online when window online event is dispatched and auto-dismisses reconnected banner', () => {
    Object.defineProperty(navigator, 'onLine', {
      value: false,
      writable: true,
      configurable: true,
    })

    render(
      <MemoryRouter>
        <OfflineBanner />
      </MemoryRouter>
    )

    expect(screen.getByTestId('offline-banner')).toBeInTheDocument()

    // Trigger online event (e.g. reconnecting to cell network)
    act(() => {
      window.dispatchEvent(new Event('online'))
    })

    // Offline banner should disappear
    expect(screen.queryByTestId('offline-banner')).not.toBeInTheDocument()

    // Reconnected banner should be displayed
    expect(screen.getByTestId('reconnected-banner')).toBeInTheDocument()
    expect(
      screen.getByText(/Connection Restored — Cloud RAG & LeafSense Sync Active/i)
    ).toBeInTheDocument()

    // Advance timers past the 3500ms timeout
    act(() => {
      vi.advanceTimersByTime(3600)
    })

    expect(screen.queryByTestId('reconnected-banner')).not.toBeInTheDocument()
  })

  it('cleans up event listeners properly upon unmounting', () => {
    const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener')

    const { unmount } = render(
      <MemoryRouter>
        <OfflineBanner />
      </MemoryRouter>
    )

    unmount()

    expect(removeEventListenerSpy).toHaveBeenCalledWith('online', expect.any(Function))
    expect(removeEventListenerSpy).toHaveBeenCalledWith('offline', expect.any(Function))

    removeEventListenerSpy.mockRestore()
  })
})
