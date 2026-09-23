import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import ErrorBoundary from './ErrorBoundary'

// Module-level flag rather than a prop: ErrorBoundary re-renders the exact
// same children element after resetError(), so the only way to make the
// "recovered" path exercise a genuinely different render is a condition
// external to props/state that changes between the throw and the retry.
let shouldThrow = true
function Bomb() {
  if (shouldThrow) throw new Error('boom')
  return <div>recovered</div>
}

describe('ErrorBoundary', () => {
  beforeEach(() => {
    shouldThrow = true
  })

  it('renders children normally when nothing throws', () => {
    render(
      <ErrorBoundary>
        <div>all good</div>
      </ErrorBoundary>
    )
    expect(screen.getByText('all good')).toBeInTheDocument()
  })

  it('catches a render error and shows the fallback UI with the error message', () => {
    // React logs caught errors to console.error by default; silence the
    // expected noise for this deliberately-throwing test.
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})

    render(
      <ErrorBoundary>
        <Bomb />
      </ErrorBoundary>
    )

    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText('boom')).toBeInTheDocument()
    expect(screen.getByText('Try Again')).toBeInTheDocument()

    consoleError.mockRestore()
  })

  it('re-renders children after "Try Again" once the underlying error condition clears', () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})

    render(
      <ErrorBoundary>
        <Bomb />
      </ErrorBoundary>
    )
    shouldThrow = false
    fireEvent.click(screen.getByText('Try Again'))

    expect(screen.getByText('recovered')).toBeInTheDocument()
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument()

    consoleError.mockRestore()
  })
})
