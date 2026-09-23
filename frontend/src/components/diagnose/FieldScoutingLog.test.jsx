import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import FieldScoutingLog from './FieldScoutingLog'
import PredictionHeroCard from './PredictionHeroCard'
import {
  SCOUTING_STORAGE_KEY,
  DEFAULT_SCOUTING_LOGS,
  getScoutingLogs,
  saveScoutingLogs,
  addScoutingEntry,
  exportScoutingToCSV,
  exportScoutingToJSON,
} from '../../utils/scoutingStorage'

const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
}))

describe('FieldScoutingLog - Field Scouting & Disease Tracking Hub', () => {
  const mockInitialLogs = [
    {
      id: 'test-scout-1',
      timestamp: '2026-08-14T14:30:00.000Z',
      crop: 'Tomato',
      disease: 'Early Blight',
      severity: 'Severe',
      confidence: 0.94,
      location: 'North Orchard Block B',
      notes: 'Concentric target ring spots detected on lower foliage.',
      remedyApplied: 'Copper Hydroxide 2.5g/L',
    },
    {
      id: 'test-scout-2',
      timestamp: '2026-08-12T09:15:00.000Z',
      crop: 'Potato',
      disease: 'Late Blight',
      severity: 'Severe',
      confidence: 0.91,
      location: 'West Field Plot 4',
      notes: 'Water-soaked lesions on leaf margins.',
      remedyApplied: 'Mancozeb 75% WP',
    },
    {
      id: 'test-scout-3',
      timestamp: '2026-08-10T16:45:00.000Z',
      crop: 'Corn',
      disease: 'Common Rust',
      severity: 'Moderate',
      confidence: 0.88,
      location: 'South Ridge Section 2',
      notes: 'Small cinnamon-brown pustules on leaves.',
      remedyApplied: 'Azoxystrobin 250 SC',
    },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    window.localStorage.setItem(SCOUTING_STORAGE_KEY, JSON.stringify(mockInitialLogs))

    globalThis.URL.createObjectURL = vi.fn(() => 'blob:http://localhost/mock-export-file')
    globalThis.URL.revokeObjectURL = vi.fn()
  })

  it('renders summary cards with correct metrics based on scouting records', () => {
    render(<FieldScoutingLog />)

    // Summary Metric Cards
    expect(screen.getByTestId('metric-total-scouts')).toHaveTextContent('3')
    expect(screen.getByTestId('metric-high-severity')).toHaveTextContent('2')
    expect(screen.getByTestId('metric-top-pathogen')).toHaveTextContent(/Early Blight|Late Blight/i)
    expect(screen.getByTestId('metric-last-scout-date')).toHaveTextContent(/Aug 14, 2026/i)

    // Header & Actions
    expect(screen.getByText('Field Scouting Log & Outbreak History')).toBeInTheDocument()
    expect(screen.getByTestId('open-add-scout-modal-button')).toBeInTheDocument()
    expect(screen.getByTestId('export-csv-button')).toBeInTheDocument()
    expect(screen.getByTestId('export-json-button')).toBeInTheDocument()
  })

  it('renders all timeline cards with crop, disease, severity, location, and remedy applied', () => {
    render(<FieldScoutingLog />)

    expect(screen.getByTestId('scouting-entry-test-scout-1')).toBeInTheDocument()
    expect(screen.getByTestId('scouting-entry-test-scout-2')).toBeInTheDocument()
    expect(screen.getByTestId('scouting-entry-test-scout-3')).toBeInTheDocument()

    // Check specific entry contents
    expect(screen.getByText('North Orchard Block B')).toBeInTheDocument()
    expect(screen.getByText('Copper Hydroxide 2.5g/L')).toBeInTheDocument()
    expect(screen.getAllByText('Early Blight').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Late Blight').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Common Rust').length).toBeGreaterThan(0)
  })

  it('filters scouting entries by crop', async () => {
    render(<FieldScoutingLog />)

    const cropSelect = screen.getByTestId('filter-crop-select')
    expect(cropSelect).toBeInTheDocument()

    // Filter to Tomato only
    fireEvent.change(cropSelect, { target: { value: 'Tomato' } })

    await waitFor(() => {
      expect(screen.getByTestId('scouting-entry-test-scout-1')).toBeInTheDocument()
      expect(screen.queryByTestId('scouting-entry-test-scout-2')).not.toBeInTheDocument()
      expect(screen.queryByTestId('scouting-entry-test-scout-3')).not.toBeInTheDocument()
    })
    expect(screen.getByText(/Showing 1 of 3 scouting records/i)).toBeInTheDocument()

    // Filter to Potato only
    fireEvent.change(cropSelect, { target: { value: 'Potato' } })
    await waitFor(() => {
      expect(screen.queryByTestId('scouting-entry-test-scout-1')).not.toBeInTheDocument()
      expect(screen.getByTestId('scouting-entry-test-scout-2')).toBeInTheDocument()
    })
  })

  it('filters scouting entries by severity level', async () => {
    render(<FieldScoutingLog />)

    const severitySelect = screen.getByTestId('filter-severity-select')

    // Filter to Moderate
    fireEvent.change(severitySelect, { target: { value: 'Moderate' } })

    await waitFor(() => {
      expect(screen.queryByTestId('scouting-entry-test-scout-1')).not.toBeInTheDocument()
      expect(screen.queryByTestId('scouting-entry-test-scout-2')).not.toBeInTheDocument()
      expect(screen.getByTestId('scouting-entry-test-scout-3')).toBeInTheDocument()
    })
  })

  it('filters entries by search query text', async () => {
    render(<FieldScoutingLog />)

    const searchInput = screen.getByTestId('search-scouting-input')
    fireEvent.change(searchInput, { target: { value: 'Orchard Block' } })

    await waitFor(() => {
      expect(screen.getByTestId('scouting-entry-test-scout-1')).toBeInTheDocument()
      expect(screen.queryByTestId('scouting-entry-test-scout-2')).not.toBeInTheDocument()
      expect(screen.queryByTestId('scouting-entry-test-scout-3')).not.toBeInTheDocument()
    })
  })

  it('adds a new scouting entry through the modal form', async () => {
    const handleEntryAdded = vi.fn()
    render(<FieldScoutingLog onEntryAdded={handleEntryAdded} />)

    // Open Modal
    const addBtn = screen.getByTestId('open-add-scout-modal-button')
    fireEvent.click(addBtn)

    await waitFor(() => {
      expect(screen.getByTestId('add-scout-modal-backdrop')).toBeInTheDocument()
    })

    // Fill form
    fireEvent.change(screen.getByTestId('form-crop-input'), { target: { value: 'Grape' } })
    fireEvent.change(screen.getByTestId('form-disease-input'), { target: { value: 'Black Rot' } })
    fireEvent.change(screen.getByTestId('form-severity-select'), { target: { value: 'Severe' } })
    fireEvent.change(screen.getByTestId('form-confidence-input'), { target: { value: '92' } })
    fireEvent.change(screen.getByTestId('form-location-input'), { target: { value: 'Vineyard Row 44' } })
    fireEvent.change(screen.getByTestId('form-remedy-input'), { target: { value: 'Myclobutanil 20EW' } })
    fireEvent.change(screen.getByTestId('form-notes-input'), { target: { value: 'Circular tan spots with dark brown margins.' } })

    // Submit form
    const submitBtn = screen.getByTestId('submit-add-scout-button')
    fireEvent.click(submitBtn)

    await waitFor(() => {
      expect(screen.queryByTestId('add-scout-modal-backdrop')).not.toBeInTheDocument()
      expect(screen.getByText('Vineyard Row 44')).toBeInTheDocument()
      expect(screen.getAllByText('Black Rot').length).toBeGreaterThan(0)
      expect(screen.getByText('Myclobutanil 20EW')).toBeInTheDocument()
    })

    expect(handleEntryAdded).toHaveBeenCalledWith(
      expect.objectContaining({
        crop: 'Grape',
        disease: 'Black Rot',
        location: 'Vineyard Row 44',
      })
    )
  })

  it('logs current diagnosis to scouting history with quick action button', async () => {
    const currentDiagnosis = {
      crop: 'Apple',
      disease: 'Cedar Apple Rust',
      confidence: 0.96,
      raw_class: 'Apple___Cedar_apple_rust',
      severity: 'Severe',
    }

    render(<FieldScoutingLog currentDiagnosis={currentDiagnosis} />)

    const logCurrentBtn = screen.getByTestId('log-current-diagnosis-button')
    expect(logCurrentBtn).toBeInTheDocument()

    fireEvent.click(logCurrentBtn)

    await waitFor(() => {
      expect(screen.getAllByText('Cedar Apple Rust').length).toBeGreaterThan(0)
      expect(screen.getByText(/Current diagnosis logged to scouting history!/i)).toBeInTheDocument()
    })
  })

  it('exports scouting history to CSV file', () => {
    render(<FieldScoutingLog />)

    const exportCsvBtn = screen.getByTestId('export-csv-button')
    fireEvent.click(exportCsvBtn)

    expect(globalThis.URL.createObjectURL).toHaveBeenCalled()
  })

  it('exports scouting history to JSON file', () => {
    render(<FieldScoutingLog />)

    const exportJsonBtn = screen.getByTestId('export-json-button')
    fireEvent.click(exportJsonBtn)

    expect(globalThis.URL.createObjectURL).toHaveBeenCalled()
  })

  it('deletes a scouting entry when delete button is clicked', async () => {
    render(<FieldScoutingLog />)

    expect(screen.getByTestId('scouting-entry-test-scout-1')).toBeInTheDocument()

    const deleteBtn = screen.getByTestId('delete-scout-test-scout-1')
    fireEvent.click(deleteBtn)

    await waitFor(() => {
      expect(screen.queryByTestId('scouting-entry-test-scout-1')).not.toBeInTheDocument()
    })
  })

  it('triggers quick save to field log from PredictionHeroCard', async () => {
    const mockDiagnosis = {
      raw_class: 'Tomato___Early_blight',
      crop: 'tomato',
      disease: 'early blight',
      confidence: 0.95,
      low_confidence: false,
    }

    const handleSaveToLog = vi.fn()

    render(
      <PredictionHeroCard
        diagnosis={mockDiagnosis}
        sessionId="session-123"
        onReset={vi.fn()}
        onSaveToLog={handleSaveToLog}
      />
    )

    const saveBtn = screen.getByTestId('save-to-field-log-button')
    expect(saveBtn).toBeInTheDocument()
    expect(saveBtn).toHaveTextContent('Save to Field Log')

    fireEvent.click(saveBtn)

    await waitFor(() => {
      expect(saveBtn).toHaveTextContent('Saved to Field Log ✓')
    })

    expect(handleSaveToLog).toHaveBeenCalledWith(
      expect.objectContaining({
        crop: 'Tomato',
        disease: 'Early blight',
      })
    )
  })
})
