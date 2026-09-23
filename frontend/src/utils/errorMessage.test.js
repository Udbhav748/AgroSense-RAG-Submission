import { describe, expect, it } from 'vitest'
import getErrorMessage, { getErrorInfo } from './errorMessage'

describe('getErrorMessage', () => {
  it('returns a friendly message for a VISION_SERVICE_ERROR regardless of the raw detail text', () => {
    const error = {
      response: { data: { detail: 'Could not reach LeafSense at http://localhost:8001: refused', error_code: 'VISION_SERVICE_ERROR' } },
    }
    expect(getErrorMessage(error)).toBe(
      "The plant diagnosis service isn't running right now. Please try again in a moment."
    )
  })

  it('formats the structured AppError shape with error_code and taxonomy_category appended', () => {
    const error = {
      response: { data: { detail: 'Rate limit exceeded', error_code: 'RATE_LIMITED', taxonomy_category: 'rate_limit' } },
    }
    expect(getErrorMessage(error)).toBe('Rate limit exceeded [RATE_LIMITED] (rate_limit)')
  })

  it('uses only the detail string when error_code/taxonomy_category are absent', () => {
    const error = { response: { data: { detail: 'Document not found' } } }
    expect(getErrorMessage(error)).toBe('Document not found')
  })

  it('extracts the first message from a FastAPI validation error array', () => {
    const error = { response: { data: { detail: [{ msg: 'field required' }, { msg: 'second error' }] } } }
    expect(getErrorMessage(error)).toBe('field required')
  })

  it('maps a client-side timeout to a fixed message', () => {
    const error = { code: 'ECONNABORTED' }
    expect(getErrorMessage(error)).toBe('The request timed out. Please try again.')
  })

  it('maps a network error to a fixed message', () => {
    const error = { message: 'Network Error' }
    expect(getErrorMessage(error)).toBe('Unable to reach the server. Please check your connection.')
  })

  it('falls back to the provided default when nothing else matches', () => {
    expect(getErrorMessage({}, 'custom fallback')).toBe('custom fallback')
    expect(getErrorMessage(undefined, 'custom fallback')).toBe('custom fallback')
  })

  it('uses the built-in default fallback when none is provided', () => {
    expect(getErrorMessage({})).toBe('Something went wrong. Please try again.')
  })
})

describe('getErrorInfo', () => {
  it('returns the structured fields when error_code is present', () => {
    const error = {
      response: { data: { detail: 'Rate limit exceeded', error_code: 'RATE_LIMITED', taxonomy_category: 'rate_limit' } },
    }
    expect(getErrorInfo(error)).toEqual({
      error_code: 'RATE_LIMITED',
      taxonomy_category: 'rate_limit',
      detail: 'Rate limit exceeded',
    })
  })

  it('returns null when there is no error_code (plain validation errors, network errors)', () => {
    expect(getErrorInfo({ response: { data: { detail: 'plain message' } } })).toBeNull()
    expect(getErrorInfo({ code: 'ECONNABORTED' })).toBeNull()
    expect(getErrorInfo(undefined)).toBeNull()
  })
})
