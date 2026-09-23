import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import PrescriptionWorkOrderModal from './PrescriptionWorkOrderModal'

describe('PrescriptionWorkOrderModal - PDF & Agronomic Work Order Generator', () => {
  const mockDiagnosis = {
    crop: 'tomato',
    disease: 'early blight',
    confidence: 0.96,
    engine: 'LeafSense Hybrid CNN+ViT (Port 8001)',
    session_id: 'sess-agri-998877',
  }

  const mockSeverity = {
    level: 'Severe',
    description: 'High pathogen dispersion rate',
  }

  const mockDosageInfo = {
    numericArea: '2500',
    unitLabel: 'sq ft',
    areaInAcres: '0.06',
    chemicalName: 'Copper Hydroxide 50 WP',
    ratePerLiter: '2.5',
    chemUnit: 'g',
    totalWaterLiters: '10.87',
    totalWaterGallons: '2.87',
    totalChemicalAmount: '27.2',
    equipmentName: 'Backpack Sprayer (15L)',
    tankCapacityLiters: '15',
    tanksRequired: '0.7',
    chemicalPerTank: '37.5',
  }

  const mockOnClose = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    window.print = vi.fn()
    globalThis.URL.createObjectURL = vi.fn(() => 'blob:http://localhost/mock-txt')
    globalThis.URL.revokeObjectURL = vi.fn()
  })

  it('renders nothing when isOpen is false', () => {
    const { container } = render(
      <PrescriptionWorkOrderModal
        isOpen={false}
        onClose={mockOnClose}
        diagnosis={mockDiagnosis}
        severity={mockSeverity}
      />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders official 5-section agronomic layout and verification seal when isOpen is true', () => {
    render(
      <PrescriptionWorkOrderModal
        isOpen={true}
        onClose={mockOnClose}
        diagnosis={mockDiagnosis}
        severity={mockSeverity}
        dosageInfo={mockDosageInfo}
      />
    )

    // Modal Document & Header
    expect(screen.getByTestId('prescription-work-order-document')).toBeInTheDocument()
    expect(screen.getAllByText(/OFFICIAL AGRONOMIC PRESCRIPTION & SPRAY WORK ORDER/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/#AGRI-88294-EXT/i).length).toBeGreaterThan(0)

    // Section 1: Field Diagnosis & Pathogen Identification
    expect(screen.getByText(/1\. Field Diagnosis & Pathogen Identification/i)).toBeInTheDocument()
    expect(screen.getAllByText(/tomato/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/early blight/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Severe/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/96%/i).length).toBeGreaterThan(0)

    // Section 2: Calculated Tank Mix & Spray Dosage Table
    expect(screen.getByText(/2\. Calculated Tank Mix & Spray Dosage Table/i)).toBeInTheDocument()
    expect(screen.getByText('Field Size')).toBeInTheDocument()
    expect(screen.getByText(/2500 sq ft/i)).toBeInTheDocument()
    expect(screen.getByText('Copper Hydroxide 50 WP')).toBeInTheDocument()
    expect(screen.getByText('Total Spray Volume')).toBeInTheDocument()
    expect(screen.getAllByText(/10\.87 Liters/i).length).toBeGreaterThan(0)
    expect(screen.getByText('Chemical Concentrate Amount')).toBeInTheDocument()
    expect(screen.getByText(/27\.2 g/i)).toBeInTheDocument()
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
    expect(screen.getAllByText(/Dr\. J\. Henderson, Ph\.D\., CCA/i).length).toBeGreaterThan(0)
    expect(screen.getByTestId('agronomist-stamp-seal')).toBeInTheDocument()
    expect(screen.getByText(/★ OFFICIAL SEAL ★/i)).toBeInTheDocument()
  })

  it('triggers PDF export and print methods when respective buttons are clicked', () => {
    render(
      <PrescriptionWorkOrderModal
        isOpen={true}
        onClose={mockOnClose}
        diagnosis={mockDiagnosis}
        severity={mockSeverity}
        dosageInfo={mockDosageInfo}
      />
    )

    // 1. Export PDF Document button in header
    const exportPdfBtn = screen.getByTestId('export-pdf-button')
    expect(exportPdfBtn).toBeInTheDocument()
    fireEvent.click(exportPdfBtn)
    expect(window.print).toHaveBeenCalledTimes(1)

    // 2. Export PDF Document button in footer
    const exportPdfFooterBtn = screen.getByTestId('export-pdf-button-footer')
    expect(exportPdfFooterBtn).toBeInTheDocument()
    fireEvent.click(exportPdfFooterBtn)
    expect(window.print).toHaveBeenCalledTimes(2)

    // 3. Print button
    const printBtn = screen.getByTestId('print-prescription-button')
    expect(printBtn).toBeInTheDocument()
    fireEvent.click(printBtn)
    expect(window.print).toHaveBeenCalledTimes(3)
  })

  it('downloads plain text work order format and allows closing modal', () => {
    render(
      <PrescriptionWorkOrderModal
        isOpen={true}
        onClose={mockOnClose}
        diagnosis={mockDiagnosis}
        severity={mockSeverity}
        dosageInfo={mockDosageInfo}
      />
    )

    const downloadTxtBtn = screen.getByTestId('download-txt-button')
    fireEvent.click(downloadTxtBtn)
    expect(globalThis.URL.createObjectURL).toHaveBeenCalled()

    const closeBtn = screen.getByLabelText('Close prescription modal')
    fireEvent.click(closeBtn)
    expect(mockOnClose).toHaveBeenCalled()
  })

  it('allows toggling safety PPE checkboxes', () => {
    render(
      <PrescriptionWorkOrderModal
        isOpen={true}
        onClose={mockOnClose}
        diagnosis={mockDiagnosis}
        severity={mockSeverity}
        dosageInfo={mockDosageInfo}
      />
    )

    const glovesLabel = screen.getByText(/Chemical-resistant nitrile \/ neoprene gloves/i).closest('label')
    expect(glovesLabel).toBeInTheDocument()
    const glovesCheckbox = glovesLabel.querySelector('input[type="checkbox"]')
    expect(glovesCheckbox.checked).toBe(true)

    fireEvent.click(glovesCheckbox)
    expect(glovesCheckbox.checked).toBe(false)
  })
})
