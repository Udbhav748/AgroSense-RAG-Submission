import { beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import Diagnose from './Diagnose'
import * as diagnoseService from '../services/diagnoseService'
import * as documentService from '../services/documentService'

const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
}))

const mockShowToast = vi.fn()
vi.mock('../hooks/useToast', () => ({
  default: () => ({ showToast: mockShowToast }),
}))

vi.mock('../services/documentService', () => ({
  getDocumentName: vi.fn((id) => (id === 'doc-agri-1' ? 'Tomato_Pathology_Guide.pdf' : null)),
}))

vi.mock('../services/diagnoseService', () => ({
  diagnoseLeaf: vi.fn(),
  diagnoseImageStream: vi.fn(),
  checkLeafSenseHealth: vi.fn(),
}))

function selectTestFile(file) {
  const dropzone = screen.getByTestId('leaf-dropzone')
  fireEvent.drop(dropzone, {
    dataTransfer: {
      files: [file],
    },
  })
}

describe('Diagnose Page - Plant Leaf Disease Diagnostic & Treatment Hub', () => {
  const mockDiagnosisResult = {
    answer: 'The leaf exhibits symptoms of Early Blight caused by Alternaria solani [1]. Immediate preventative fungicide or bio-remedies are recommended.',
    sources: [
      {
        chunk_id: 'chunk-agri-1',
        document_id: 'doc-agri-1',
        number: 1,
        page_number: 12,
        excerpt: 'Alternaria solani produces characteristic target-board concentric ring spots on tomato leaves.',
      },
    ],
    processing_time: 1.28,
    session_id: 'session-agri-test-123',
    diagnosis: {
      raw_class: 'Tomato___Early_blight',
      crop: 'tomato',
      disease: 'early blight',
      confidence: 0.95,
      low_confidence: false,
      heatmap_base64: 'fake_heatmap_base64_png_payload',
      infected_area_percentage: 24.5,
      lesion_count: 5,
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    window.print = vi.fn()
    globalThis.URL.createObjectURL = vi.fn(() => 'blob:http://localhost/mock-leaf-preview')
    globalThis.URL.revokeObjectURL = vi.fn()
    diagnoseService.checkLeafSenseHealth.mockResolvedValue({
      online: true,
      port: 8001,
      url: 'http://localhost:8001',
      status: 'ok',
    })
    diagnoseService.diagnoseLeaf.mockResolvedValue(mockDiagnosisResult)
    diagnoseService.diagnoseImageStream.mockImplementation(async (file, query, onEvent) => {
      onEvent?.({
        type: 'diagnosis',
        diagnosis: mockDiagnosisResult.diagnosis,
      })
      onEvent?.({
        type: 'answer_chunk',
        text: mockDiagnosisResult.answer,
      })
      onEvent?.({
        type: 'done',
        payload: mockDiagnosisResult,
      })
    })
  })

  it('renders header, service health indicator, and upload dropzone', async () => {
    render(<Diagnose />)

    expect(screen.getByText('Plant Leaf Disease Diagnostic & Treatment Hub')).toBeInTheDocument()
    expect(
      screen.getByText(/Upload or capture a leaf photo/i)
    ).toBeInTheDocument()

    // Service health indicator online
    await waitFor(() => {
      expect(screen.getByText(/LeafSense Vision Engine Active/i)).toBeInTheDocument()
      expect(screen.getByText('Port 8001')).toBeInTheDocument()
    })

    // Action buttons
    expect(screen.getByRole('button', { name: 'Upload leaf photo' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Take leaf photo' })).toBeInTheDocument()
    expect(screen.getByText('Choose from Gallery / Files')).toBeInTheDocument()
  })

  it('handles image selection, shows preview, and handles image removal', async () => {
    render(<Diagnose />)

    const file = new File(['fake-leaf-bytes'], 'tomato_early_blight.jpg', { type: 'image/jpeg' })
    selectTestFile(file)

    // Preview should now appear
    await waitFor(() => {
      expect(screen.getByText('tomato_early_blight.jpg')).toBeInTheDocument()
      const previewImg = screen.getByAltText('Plant leaf preview')
      expect(previewImg).toBeInTheDocument()
      expect(previewImg).toHaveAttribute('src', 'blob:http://localhost/mock-leaf-preview')
    })

    // Context input should be present
    expect(screen.getByPlaceholderText(/Add any specific context, crop stage, or question/i)).toBeInTheDocument()

    // Remove photo
    const removeBtn = screen.getByRole('button', { name: 'Remove image' })
    fireEvent.click(removeBtn)

    // Should return to dropzone
    await waitFor(() => {
      expect(screen.queryByAltText('Plant leaf preview')).not.toBeInTheDocument()
      expect(screen.getByText('Choose from Gallery / Files')).toBeInTheDocument()
    })
  })

  it('submits leaf photo, displays analyzing state, and renders prediction hero card with memory bridge', async () => {
    render(<Diagnose />)

    const file = new File(['fake-leaf-bytes'], 'tomato_leaf.png', { type: 'image/png' })
    selectTestFile(file)

    await waitFor(() => {
      expect(screen.getByAltText('Plant leaf preview')).toBeInTheDocument()
    })

    // Add optional context
    const contextInput = screen.getByPlaceholderText(/Add any specific context/i)
    fireEvent.change(contextInput, { target: { value: 'Observed yellow halos on bottom leaves' } })

    // Click Diagnose button
    const diagnoseBtn = screen.getByRole('button', { name: /Diagnose Plant Leaf/i })
    fireEvent.click(diagnoseBtn)

    expect(diagnoseService.diagnoseImageStream).toHaveBeenCalledWith(
      file,
      expect.objectContaining({
        query: 'Observed yellow halos on bottom leaves',
        engine: 'hybrid',
      }),
      expect.any(Function),
      expect.any(Function)
    )

    // Wait for Hero Card to render
    await waitFor(() => {
      expect(screen.getByText('early blight')).toBeInTheDocument()
    })

    // Validate Prediction Hero Card details
    expect(screen.getByText('tomato')).toBeInTheDocument()
    expect(screen.getByTestId('severity-badge')).toHaveTextContent(/Severity:\s*(Severe|Moderate)/i)
    expect(screen.getByTestId('confidence-score-value')).toHaveTextContent('95%')
    expect(screen.getByText('Tomato___Early_blight')).toBeInTheDocument()

    // Action buttons
    const chatBtn = screen.getByTestId('continue-in-chat-button')
    expect(chatBtn).toBeInTheDocument()
    expect(screen.getByTestId('download-prescription-button')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Diagnose Another Leaf/i })).toBeInTheDocument()

    // Test navigating to chat with URL search parameters and state memory bridge
    fireEvent.click(chatBtn)
    expect(mockNavigate).toHaveBeenCalledWith(
      '/chat?crop=tomato&disease=early+blight&severity=Severe&lang=en',
      expect.objectContaining({
        state: expect.objectContaining({
          sessionId: 'session-agri-test-123',
          crop: 'tomato',
          disease: 'early blight',
          language: 'en',
        }),
      })
    )
  })

  it('displays OOD Non-Leaf Gatekeeper warning banner when confidence is low or flagged uncertain', async () => {
    const oodResult = {
      ...mockDiagnosisResult,
      diagnosis: {
        raw_class: 'background_non_leaf',
        crop: 'non-leaf',
        disease: 'uncertain condition',
        confidence: 0.34,
        low_confidence: true,
        is_non_leaf: true,
      },
    }

    diagnoseService.diagnoseImageStream.mockImplementationOnce(async (file, query, onEvent) => {
      onEvent?.({
        type: 'diagnosis',
        diagnosis: oodResult.diagnosis,
      })
      onEvent?.({
        type: 'answer_chunk',
        text: oodResult.answer,
      })
      onEvent?.({
        type: 'done',
        payload: oodResult,
      })
    })

    render(<Diagnose />)

    const file = new File(['fake-random-object-bytes'], 'random_desk_photo.jpg', { type: 'image/jpeg' })
    selectTestFile(file)

    await waitFor(() => {
      expect(screen.getByAltText('Plant leaf preview')).toBeInTheDocument()
    })

    const diagnoseBtn = screen.getByRole('button', { name: /Diagnose Plant Leaf/i })
    fireEvent.click(diagnoseBtn)

    await waitFor(() => {
      expect(
        screen.getAllByText(/Low Visual Confidence \/ Non-Plant Detected/i).length
      ).toBeGreaterThan(0)
    })

    expect(
      screen.getAllByText(/Please ensure the photo is well-lit and clearly shows an infected crop leaf/i).length
    ).toBeGreaterThan(0)
  })

  it('progressively streams diagnosis: immediately displays hero card on diagnosis event, streams tokens, and finalizes on done', async () => {
    let emitEvent
    diagnoseService.diagnoseImageStream.mockImplementation(async (file, query, onEvent) => {
      emitEvent = onEvent
    })

    render(<Diagnose />)

    const file = new File(['fake-leaf-bytes'], 'tomato_early_blight.png', { type: 'image/png' })
    selectTestFile(file)

    await waitFor(() => {
      expect(screen.getByAltText('Plant leaf preview')).toBeInTheDocument()
    })

    const diagnoseBtn = screen.getByRole('button', { name: /Diagnose Plant Leaf/i })
    fireEvent.click(diagnoseBtn)

    // 1. Sub-second diagnosis event arrives immediately
    act(() => {
      emitEvent({
        type: 'diagnosis',
        diagnosis: {
          raw_class: 'Tomato___Early_blight',
          crop: 'tomato',
          disease: 'early blight',
          confidence: 0.95,
          low_confidence: false,
        },
      })
    })

    // Prediction Hero Card renders immediately
    await waitFor(() => {
      expect(screen.getByText('early blight')).toBeInTheDocument()
    })
    expect(screen.getByText('tomato')).toBeInTheDocument()
    expect(screen.getByTestId('severity-badge')).toHaveTextContent(/Severity:\s*(Severe|Moderate)/i)
    expect(screen.getByTestId('confidence-score-value')).toHaveTextContent('95%')
    expect(screen.getByText(/Streaming response\.\.\./i)).toBeInTheDocument()

    // 2. Stream tokens arrive in real-time
    act(() => {
      emitEvent({
        type: 'answer_chunk',
        text: 'Early Blight initial symptoms detected. ',
      })
    })

    await waitFor(() => {
      expect(screen.getByText(/Early Blight initial symptoms detected/i)).toBeInTheDocument()
    })

    act(() => {
      emitEvent({
        type: 'answer_chunk',
        text: 'Apply Copper Hydroxide spray immediately.',
      })
    })

    await waitFor(() => {
      expect(
        screen.getByText(/Early Blight initial symptoms detected\. Apply Copper Hydroxide spray immediately\./i)
      ).toBeInTheDocument()
    })

    // User can switch tabs while streaming
    const tabOrganic = screen.getByRole('tab', { name: /Organic Remedies/i })
    fireEvent.click(tabOrganic)
    await waitFor(() => {
      expect(screen.getByText(/Biological Fungicides & Antagonists/i)).toBeInTheDocument()
    })

    // Switch back to overview tab
    const tabOverview = screen.getByRole('tab', { name: /Overview & Symptoms/i })
    fireEvent.click(tabOverview)

    // 3. Done event arrives and finalizes
    act(() => {
      emitEvent({
        type: 'done',
        payload: {
          ...mockDiagnosisResult,
          answer: 'Early Blight initial symptoms detected. Apply Copper Hydroxide spray immediately.',
        },
      })
    })

    await waitFor(() => {
      expect(screen.queryByText(/Streaming response\.\.\./i)).not.toBeInTheDocument()
    })
  })

  it('filters active chemical ingredients dynamically based on regional regulatory jurisdiction (EFSA & OMRI)', async () => {
    diagnoseService.diagnoseLeaf.mockResolvedValueOnce(mockDiagnosisResult)

    render(<Diagnose />)

    const file = new File(['fake-bytes'], 'tomato_leaf.png', { type: 'image/png' })
    selectTestFile(file)

    await waitFor(() => {
      expect(screen.getByAltText('Plant leaf preview')).toBeInTheDocument()
    })

    const diagnoseBtn = screen.getByRole('button', { name: /Diagnose Plant Leaf/i })
    fireEvent.click(diagnoseBtn)

    await waitFor(() => {
      expect(screen.getByText('early blight')).toBeInTheDocument()
    })

    // Navigate to Tab 3: Chemical Control & Dosages
    const tabChemical = screen.getByRole('tab', { name: /Chemical Control & Dosages/i })
    fireEvent.click(tabChemical)

    await waitFor(() => {
      expect(screen.getByTestId('regulatory-jurisdiction-select')).toBeInTheDocument()
    })

    const jurisdictionSelect = screen.getByTestId('regulatory-jurisdiction-select')

    // 1. Select European Union (EFSA)
    fireEvent.change(jurisdictionSelect, { target: { value: 'EFSA' } })

    await waitFor(() => {
      expect(screen.getByText(/Non-Renewed in EU \/ Restricted/i)).toBeInTheDocument()
      expect(screen.getByText(/Phase-out in EU/i)).toBeInTheDocument()
    })

    // Approved synthetic like Azoxystrobin is compliant under EFSA
    expect(screen.getAllByText(/EFSA Compliant/i).length).toBeGreaterThan(0)

    // 2. Select Global Organic (OMRI Only)
    fireEvent.change(jurisdictionSelect, { target: { value: 'OMRI' } })

    await waitFor(() => {
      expect(screen.getAllByText(/Prohibited \(OMRI Organic\)/i).length).toBeGreaterThan(0)
      expect(screen.getByText(/Global Organic \(OMRI Only\) Active/i)).toBeInTheDocument()
    })
  })

  it('opens and formats downloadable/printable spray prescription work order with PDF export, PPE, REI/PHI, and Agronomist verification', async () => {
    diagnoseService.diagnoseLeaf.mockResolvedValueOnce(mockDiagnosisResult)

    render(<Diagnose />)

    const file = new File(['fake-bytes'], 'tomato_leaf.png', { type: 'image/png' })
    selectTestFile(file)

    await waitFor(() => {
      expect(screen.getByAltText('Plant leaf preview')).toBeInTheDocument()
    })

    const diagnoseBtn = screen.getByRole('button', { name: /Diagnose Plant Leaf/i })
    fireEvent.click(diagnoseBtn)

    await waitFor(() => {
      expect(screen.getByTestId('download-prescription-button')).toBeInTheDocument()
    })

    // Click Download Spray Prescription Work Order button
    const prescriptionBtn = screen.getByTestId('download-prescription-button')
    fireEvent.click(prescriptionBtn)

    // Modal should be open
    await waitFor(() => {
      expect(screen.getByTestId('prescription-work-order-document')).toBeInTheDocument()
    })

    // Verify Header
    expect(screen.getAllByText(/OFFICIAL AGRONOMIC PRESCRIPTION & SPRAY WORK ORDER/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/#AGRI-88294-EXT/i).length).toBeGreaterThan(0)

    // Section 1: Field Diagnosis & Pathogen Identification
    expect(screen.getByText(/1\. Field Diagnosis & Pathogen Identification/i)).toBeInTheDocument()
    expect(screen.getAllByText(/tomato/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/early blight/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/95%/i).length).toBeGreaterThan(0)

    // Section 2: Calculated Tank Mix & Spray Dosage Table
    expect(screen.getByText(/2\. Calculated Tank Mix & Spray Dosage Table/i)).toBeInTheDocument()
    expect(screen.getByText('Field Size')).toBeInTheDocument()
    expect(screen.getByText('Total Spray Volume')).toBeInTheDocument()
    expect(screen.getByText('Chemical Concentrate Amount')).toBeInTheDocument()
    expect(screen.getByText('Water Carrier Volume')).toBeInTheDocument()

    // Section 3: Worker Protection Standard (WPS) & Safety PPE
    expect(screen.getByText(/3\. Worker Protection Standard \(WPS\) & Safety PPE/i)).toBeInTheDocument()
    expect(screen.getByText(/Chemical-resistant nitrile \/ neoprene gloves \(Nitrile gloves\)/i)).toBeInTheDocument()
    expect(screen.getByText(/N95 \/ organic vapor respirator mask \(N95\/organic vapor respirator\)/i)).toBeInTheDocument()
    expect(screen.getByText(/Protective chemical splash goggles \/ face shield \(Splash goggles\)/i)).toBeInTheDocument()
    expect(screen.getByText(/Long-sleeved chemical coveralls \/ chemical apron & waterproof rubber boots \(Chemical apron\)/i)).toBeInTheDocument()

    // Section 4: Mandatory Re-Entry Interval (REI: 12-24h) & Pre-Harvest Interval (PHI: 0-7d)
    expect(screen.getByText(/4\. Mandatory Re-Entry Interval \(REI: 12-24h\) & Pre-Harvest Interval \(PHI: 0-7d\)/i)).toBeInTheDocument()
    expect(screen.getByText(/Restricted Entry Interval \(REI\): 12 - 24 Hours \(12-24h\)/i)).toBeInTheDocument()
    expect(screen.getByText(/Pre-Harvest Interval \(PHI\): 0 - 7 Days \(0-7d\)/i)).toBeInTheDocument()

    // Section 5: Agronomist Certification & Stamp Seal
    expect(screen.getByText(/5\. Agronomist Certification & Stamp Seal/i)).toBeInTheDocument()
    expect(screen.getByText(/Certified Agronomist:/i)).toBeInTheDocument()
    expect(screen.getAllByText(/Dr\. J\. Henderson, Ph\.D\., CCA/i).length).toBeGreaterThan(0)
    expect(screen.getByTestId('agronomist-stamp-seal')).toBeInTheDocument()
    expect(screen.getByText(/★ OFFICIAL SEAL ★/i)).toBeInTheDocument()
    expect(screen.getByText(/Authorized Agronomist Signature/i)).toBeInTheDocument()

    // Test Export PDF Document Button Trigger
    const exportPdfButtons = screen.getAllByRole('button', { name: /📥 Export PDF Document/i })
    expect(exportPdfButtons.length).toBeGreaterThan(0)
    fireEvent.click(exportPdfButtons[0])
    expect(window.print).toHaveBeenCalled()

    // Test Print Button Trigger
    const printBtn = screen.getByTestId('print-prescription-button')
    expect(printBtn).toBeInTheDocument()
    fireEvent.click(printBtn)
    expect(window.print).toHaveBeenCalledTimes(2)

    // Test Download (.txt) action
    const downloadTxtBtn = screen.getByTestId('download-txt-button')
    expect(downloadTxtBtn).toBeInTheDocument()
    fireEvent.click(downloadTxtBtn)
    expect(globalThis.URL.createObjectURL).toHaveBeenCalled()

    // Test Close Modal
    const closeBtn = screen.getByLabelText(/Close prescription modal/i)
    fireEvent.click(closeBtn)

    await waitFor(() => {
      expect(screen.queryByTestId('prescription-work-order-document')).not.toBeInTheDocument()
    })
  })

  it('switches between all 5 treatment tabs and renders structured agronomic content', async () => {
    diagnoseService.diagnoseLeaf.mockResolvedValueOnce(mockDiagnosisResult)

    render(<Diagnose />)

    const file = new File(['fake-bytes'], 'tomato_leaf.png', { type: 'image/png' })
    selectTestFile(file)

    await waitFor(() => {
      expect(screen.getByAltText('Plant leaf preview')).toBeInTheDocument()
    })

    const diagnoseBtn = screen.getByRole('button', { name: /Diagnose Plant Leaf/i })
    fireEvent.click(diagnoseBtn)

    await waitFor(() => {
      expect(screen.getByText('early blight')).toBeInTheDocument()
    })

    // Verify all 5 tab buttons exist
    const tabOverview = screen.getByRole('tab', { name: /Overview & Symptoms/i })
    const tabOrganic = screen.getByRole('tab', { name: /Organic Remedies/i })
    const tabChemical = screen.getByRole('tab', { name: /Chemical Control & Dosages/i })
    const tabPrevention = screen.getByRole('tab', { name: /Prevention Schedule/i })
    const tabSources = screen.getByRole('tab', { name: /Verified Sources & PDF Citations/i })

    expect(tabOverview).toBeInTheDocument()
    expect(tabOrganic).toBeInTheDocument()
    expect(tabChemical).toBeInTheDocument()
    expect(tabPrevention).toBeInTheDocument()
    expect(tabSources).toBeInTheDocument()

    // 1. Tab 1: Overview & Symptoms (active by default)
    expect(screen.getByText(/Primary Diagnostic Symptoms/i)).toBeInTheDocument()
    expect(screen.getByText(/concentric rings/i)).toBeInTheDocument()

    // 2. Tab 2: Organic Remedies
    fireEvent.click(tabOrganic)
    await waitFor(() => {
      expect(screen.getByText(/Biological Fungicides & Antagonists/i)).toBeInTheDocument()
    })
    expect(screen.getAllByText(/Bacillus subtilis/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Neem Oil/i).length).toBeGreaterThan(0)

    // 3. Tab 3: Chemical Control & Dosages
    fireEvent.click(tabChemical)
    await waitFor(() => {
      expect(screen.getByText(/Recommended Active Chemical Ingredients & Dosages/i)).toBeInTheDocument()
    })
    expect(screen.getAllByText(/Chlorothalonil 720 SC/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Mancozeb 75% WP/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/FRAC Resistance Management/i)).toBeInTheDocument()

    // 4. Tab 4: Prevention Schedule
    fireEvent.click(tabPrevention)
    await waitFor(() => {
      expect(screen.getByText(/4-Stage Seasonal Crop Protection Calendar/i)).toBeInTheDocument()
    })
    expect(screen.getByText('Pre-Planting')).toBeInTheDocument()
    expect(screen.getByText('Vegetative Growth')).toBeInTheDocument()
    expect(screen.getByText('Flowering & Fruiting')).toBeInTheDocument()
    expect(screen.getByText('Post-Harvest Cleanup')).toBeInTheDocument()

    // 5. Tab 5: Verified Sources & PDF Citations
    fireEvent.click(tabSources)
    await waitFor(() => {
      expect(screen.getByText(/RAG Retrieved Documents & Citations/i)).toBeInTheDocument()
    })
    expect(screen.getAllByText(/Tomato_Pathology_Guide\.pdf · page 12/i).length).toBeGreaterThan(0)
  })

  it('performs interactive spray dosage calculations when field size and units change', async () => {
    diagnoseService.diagnoseLeaf.mockResolvedValueOnce(mockDiagnosisResult)

    render(<Diagnose />)

    const file = new File(['fake-bytes'], 'tomato_leaf.png', { type: 'image/png' })
    selectTestFile(file)

    await waitFor(() => {
      expect(screen.getByAltText('Plant leaf preview')).toBeInTheDocument()
    })

    const diagnoseBtn = screen.getByRole('button', { name: /Diagnose Plant Leaf/i })
    fireEvent.click(diagnoseBtn)

    await waitFor(() => {
      expect(screen.getByText('Field Spray Dosage & Tank Mix Calculator')).toBeInTheDocument()
    })

    const fieldSizeInput = screen.getByTestId('field-size-input')
    const unitSelect = screen.getByTestId('area-unit-select')
    const chemicalSelect = screen.getByTestId('chemical-preset-select')

    expect(fieldSizeInput).toBeInTheDocument()
    expect(unitSelect).toBeInTheDocument()

    // Default: 1000 sq ft, Copper Hydroxide (2.5 g/L)
    const initialLiters = screen.getByTestId('water-volume-liters')
    expect(initialLiters).toHaveTextContent(/4\.35\s*L/i)
    expect(screen.getByTestId('chemical-amount-output')).toHaveTextContent(/10\.9\s*g/i)

    // Change Field Size to 5000 sq ft
    fireEvent.change(fieldSizeInput, { target: { value: '5000' } })
    expect(screen.getByTestId('water-volume-liters')).toHaveTextContent(/21\.73\s*L/i)
    expect(screen.getByTestId('chemical-amount-output')).toHaveTextContent(/54\.3\s*g/i)

    // Change Unit to Acres (2 acres)
    fireEvent.change(unitSelect, { target: { value: 'acres' } })
    fireEvent.change(fieldSizeInput, { target: { value: '2' } })

    expect(screen.getByTestId('water-volume-liters')).toHaveTextContent(/378\.54\s*L/i)
    expect(screen.getByTestId('chemical-amount-output')).toHaveTextContent(/946\.4\s*g/i)
    expect(screen.getByTestId('tank-refills-count')).toHaveTextContent(/~23\.7\s*tanks/i)

    // Change Chemical to Neem Oil (5 ml/L)
    fireEvent.change(chemicalSelect, { target: { value: 'neem' } })
    expect(screen.getByTestId('chemical-amount-output')).toHaveTextContent(/1892\.7\s*ml/i)
  })

  it('displays LeafSense offline error banner with actionable instructions when service is down', async () => {
    diagnoseService.checkLeafSenseHealth.mockResolvedValue({
      online: false,
      port: 8001,
      url: 'http://localhost:8001',
      message: 'Connection refused',
    })

    render(<Diagnose />)

    await waitFor(() => {
      expect(screen.getByTestId('leafsense-offline-banner')).toBeInTheDocument()
    })

    expect(screen.getByText('LeafSense Vision Service Offline (Port 8001)')).toBeInTheDocument()
    expect(screen.getByText(/uvicorn main:app --host localhost --port 8001/i)).toBeInTheDocument()
    expect(screen.getByText(/\.\/start-local\.ps1/i)).toBeInTheDocument()

    // Test Recheck button
    diagnoseService.checkLeafSenseHealth.mockResolvedValue({
      online: true,
      port: 8001,
      url: 'http://localhost:8001',
      status: 'ok',
    })

    const recheckBtn = screen.getByRole('button', { name: /Recheck Port 8001/i })
    fireEvent.click(recheckBtn)

    await waitFor(() => {
      expect(screen.getByText(/LeafSense Vision Engine Active/i)).toBeInTheDocument()
    })
  })

  it('provides hands-free speech-to-text dictation button for field context input', async () => {
    render(<Diagnose />)

    const file = new File(['fake-bytes'], 'tomato_leaf.png', { type: 'image/png' })
    selectTestFile(file)

    await waitFor(() => {
      expect(screen.getByAltText('Plant leaf preview')).toBeInTheDocument()
    })

    // Microphone button should be rendered next to context input
    const micButton = screen.getByTestId('speech-to-text-mic-button')
    expect(micButton).toBeInTheDocument()
    expect(micButton).toHaveAttribute('aria-label', 'Start hands-free voice dictation')

    // Click microphone to toggle listening
    fireEvent.click(micButton)

    // Status pill should appear
    await waitFor(() => {
      expect(screen.getByText(/Listening\.\.\. Speak clearly/i)).toBeInTheDocument()
    })
  })

  it('renders emergency 24h field protocol audio player and triggers speech synthesis voice narration with playback controls', async () => {
    diagnoseService.diagnoseLeaf.mockResolvedValueOnce(mockDiagnosisResult)

    render(<Diagnose />)

    const file = new File(['fake-bytes'], 'tomato_leaf.png', { type: 'image/png' })
    selectTestFile(file)

    await waitFor(() => {
      expect(screen.getByAltText('Plant leaf preview')).toBeInTheDocument()
    })

    const diagnoseBtn = screen.getByRole('button', { name: /Diagnose Plant Leaf/i })
    fireEvent.click(diagnoseBtn)

    await waitFor(() => {
      expect(screen.getByText('early blight')).toBeInTheDocument()
    })

    // Verify 24h Field Protocol audio player is present
    expect(screen.getByTestId('field-protocol-audio-player')).toBeInTheDocument()
    expect(screen.getByTestId('audio-soundwave-bars')).toBeInTheDocument()

    const playBtn = screen.getByTestId('voice-play-protocol-button')
    expect(playBtn).toBeInTheDocument()
    expect(playBtn).toHaveTextContent(/Listen to 24h Field Protocol/i)

    // Click play button to start speech synthesis
    fireEvent.click(playBtn)

    expect(window.speechSynthesis.speak).toHaveBeenCalledWith(
      expect.objectContaining({
        text: expect.stringMatching(/Emergency 24 to 48-Hour Field Protocol.*tomato.*early blight/i),
      })
    )

    // Verify Pause button is rendered and clickable
    await waitFor(() => {
      expect(screen.getByTestId('voice-pause-protocol-button')).toBeInTheDocument()
    })

    const pauseBtn = screen.getByTestId('voice-pause-protocol-button')
    fireEvent.click(pauseBtn)
    expect(window.speechSynthesis.pause).toHaveBeenCalled()

    // Resume playback
    const resumeBtn = screen.getByTestId('voice-play-protocol-button')
    expect(resumeBtn).toHaveTextContent('Resume')
    fireEvent.click(resumeBtn)
    expect(window.speechSynthesis.resume).toHaveBeenCalled()

    // Stop playback
    const stopBtn = screen.getByTestId('voice-stop-protocol-button')
    fireEvent.click(stopBtn)
    expect(window.speechSynthesis.cancel).toHaveBeenCalled()
  })

  it('renders visual metric badges (Infected Leaf Area & Estimated Lesions) and interactive heatmap overlay toggle with opacity slider', async () => {
    diagnoseService.diagnoseLeaf.mockResolvedValueOnce(mockDiagnosisResult)

    render(<Diagnose />)

    const file = new File(['fake-bytes'], 'tomato_leaf.png', { type: 'image/png' })
    selectTestFile(file)

    await waitFor(() => {
      expect(screen.getByAltText('Plant leaf preview')).toBeInTheDocument()
    })

    const diagnoseBtn = screen.getByRole('button', { name: /Diagnose Plant Leaf/i })
    fireEvent.click(diagnoseBtn)

    await waitFor(() => {
      expect(screen.getByText('early blight')).toBeInTheDocument()
    })

    // 1. Verify Visual Metric Badges
    const infectedAreaBadge = screen.getByTestId('infected-area-badge')
    expect(infectedAreaBadge).toBeInTheDocument()
    expect(infectedAreaBadge).toHaveTextContent(/Infected Leaf Area:\s*24\.5%/i)

    const lesionCountBadge = screen.getByTestId('lesion-count-badge')
    expect(lesionCountBadge).toBeInTheDocument()
    expect(lesionCountBadge).toHaveTextContent(/Estimated Lesions:\s*5\s*spots/i)

    // 2. Verify Heatmap Overlay & Original Photo Toggle Buttons
    const toggleOriginalBtn = screen.getByTestId('toggle-original-photo')
    const toggleHeatmapBtn = screen.getByTestId('toggle-heatmap-overlay')
    expect(toggleOriginalBtn).toBeInTheDocument()
    expect(toggleHeatmapBtn).toBeInTheDocument()

    // 3. Verify Opacity Slider
    const opacitySlider = screen.getByTestId('heatmap-opacity-slider')
    expect(opacitySlider).toBeInTheDocument()
    expect(opacitySlider).toHaveValue('80')

    // Change Opacity Slider value to 45%
    fireEvent.change(opacitySlider, { target: { value: '45' } })
    expect(opacitySlider).toHaveValue('45')
    expect(screen.getByText('45%')).toBeInTheDocument()

    // Switch to Original Photo mode
    fireEvent.click(toggleOriginalBtn)
    expect(screen.getByText('0%')).toBeInTheDocument()
    expect(opacitySlider).toBeDisabled()

    // Switch back to Heatmap Lesion Overlay mode
    fireEvent.click(toggleHeatmapBtn)
    expect(screen.getByText('45%')).toBeInTheDocument()
    expect(opacitySlider).not.toBeDisabled()

    // Verify Color Legend items
    expect(screen.getByText('Healthy Tissue')).toBeInTheDocument()
    expect(screen.getByText('Chlorotic Margins')).toBeInTheDocument()
    expect(screen.getByText('Active Lesion Centers')).toBeInTheDocument()
  })

  it('handles language selector changes and assigns appropriate STT and TTS voice codes', async () => {
    let capturedRecognitionInstance = null
    class MockSpeechRecognition {
      constructor() {
        this.lang = 'en-US'
        this.continuous = false
        this.interimResults = false
        this.onresult = null
        this.onerror = null
        this.onend = null
        // eslint-disable-next-line consistent-this
        capturedRecognitionInstance = this
      }
      start() {
        if (this.onresult) {
          this.onresult({
            resultIndex: 0,
            results: [[{ transcript: 'Hojas amarillas con manchas negras' }]],
          })
        }
      }
      stop() {
        if (this.onend) this.onend()
      }
      abort() {}
    }

    window.SpeechRecognition = MockSpeechRecognition
    window.webkitSpeechRecognition = MockSpeechRecognition

    render(<Diagnose />)

    const file = new File(['fake-bytes'], 'tomato_leaf.png', { type: 'image/png' })
    selectTestFile(file)

    await waitFor(() => {
      expect(screen.getByAltText('Plant leaf preview')).toBeInTheDocument()
    })

    // 1. Language selector in upload preview area
    const langSelect = screen.getByTestId('voice-language-selector')
    expect(langSelect).toBeInTheDocument()
    expect(langSelect).toHaveValue('en')

    // Change language to Spanish (es)
    fireEvent.change(langSelect, { target: { value: 'es' } })
    expect(langSelect).toHaveValue('es')

    // 2. Click mic button to test Speech Recognition language code assignment
    const micButton = screen.getByTestId('speech-to-text-mic-button')
    fireEvent.click(micButton)

    expect(capturedRecognitionInstance).not.toBeNull()
    expect(capturedRecognitionInstance.lang).toBe('es-ES')

    // Context textarea should have received transcribed text
    const queryInput = screen.getByLabelText(/Optional context or symptoms observed/i)
    expect(queryInput).toHaveValue('Hojas amarillas con manchas negras')

    // 3. Diagnose with Spanish language parameter
    const diagnoseBtn = screen.getByRole('button', { name: /Diagnose Plant Leaf/i })
    fireEvent.click(diagnoseBtn)

    await waitFor(() => {
      expect(diagnoseService.diagnoseImageStream).toHaveBeenCalledWith(
        file,
        expect.objectContaining({
          language: 'es',
        }),
        expect.any(Function),
        expect.any(Function)
      )
    })

    // 4. Verify TTS Audio Player in Diagnosis Result
    await waitFor(() => {
      expect(screen.getByTestId('voice-play-protocol-button')).toBeInTheDocument()
    })

    let capturedUtterance = null
    window.speechSynthesis.speak = vi.fn((utterance) => {
      capturedUtterance = utterance
    })

    const playBtn = screen.getByTestId('voice-play-protocol-button')
    fireEvent.click(playBtn)

    expect(window.speechSynthesis.speak).toHaveBeenCalled()
    expect(capturedUtterance).not.toBeNull()
    expect(capturedUtterance.lang).toBe('es-ES')
    expect(capturedUtterance.text).toContain('Protocolo de Campo')
  })
})
