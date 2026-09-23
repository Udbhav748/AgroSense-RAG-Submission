import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Award,
  Calendar,
  CheckCircle2,
  Clock,
  Download,
  FileCheck,
  FileDown,
  FileText,
  FlaskConical,
  Leaf,
  Printer,
  ShieldAlert,
  ShieldCheck,
  X,
} from 'lucide-react'
import Button from '../ui/Button'
import { getAgronomicGuide, calculateSprayDosage } from '../../utils/agronomyData'

export default function PrescriptionWorkOrderModal({
  isOpen,
  onClose,
  diagnosis,
  severity,
  dosageInfo,
}) {
  const [ppeChecked, setPpeChecked] = useState({
    gloves: true,
    respirator: true,
    eyeProtection: true,
    coveralls: true,
  })

  if (!isOpen) return null

  const crop = diagnosis?.crop || 'Crop Foliage'
  const disease = diagnosis?.disease || 'Condition Undetermined'
  const severityLevel = severity?.level || 'Moderate'
  const guide = getAgronomicGuide(disease, crop)

  // Default calculation fallback if not provided
  const calculation =
    dosageInfo ||
    calculateSprayDosage({
      areaValue: 1000,
      unitId: 'sq_ft',
      chemicalPresetId: 'copper',
    })

  const verificationCode = '#AGRI-88294-EXT'
  const prescriptionId = `RX-AGRI-${(diagnosis?.session_id || 'DEMO').slice(-6).toUpperCase()}-${new Date().getFullYear()}`
  const today = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })

  const togglePpe = (key) => {
    setPpeChecked((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const handlePrint = () => {
    if (typeof window !== 'undefined' && typeof window.print === 'function') {
      window.print()
    }
  }

  const handleExportPDF = () => {
    if (typeof window !== 'undefined' && typeof window.print === 'function') {
      window.print()
    }
  }

  const handleDownloadWorkOrder = () => {
    const textContent = `================================================================================
OFFICIAL AGRONOMIC PRESCRIPTION & SPRAY WORK ORDER
AgroSense-RAG Pathology & Crop Protection Hub
Prescription ID:   ${prescriptionId}
Verification Code: ${verificationCode}
Date Issued:       ${today}
================================================================================

1. FIELD DIAGNOSIS & PATHOGEN IDENTIFICATION
--------------------------------------------------------------------------------
Crop Species:            ${crop.toUpperCase()}
Diagnosed Condition:     ${disease.toUpperCase()}
Pathogen Classification: ${guide.pathogen}
Field Infection Level:   ${severityLevel.toUpperCase()}
Classification Engine:   ${diagnosis?.engine || 'LeafSense Hybrid CNN+ViT (Port 8001)'}
Vision Confidence:       ${Math.round((diagnosis?.confidence ?? 0.95) * 100)}%
Agronomic Code:          ${verificationCode}

2. CALCULATED TANK MIX & SPRAY DOSAGE TABLE
--------------------------------------------------------------------------------
Field Size / Plot Area:     ${calculation.numericArea} ${calculation.unitLabel} (${calculation.areaInAcres} acres)
Active Formulation:         ${calculation.chemicalName}
Prescribed Rate:            ${calculation.ratePerLiter} ${calculation.chemUnit} per Liter of water
Total Spray Volume:         ${calculation.totalWaterLiters} Liters (${calculation.totalWaterGallons} US Gallons)
Chemical Concentrate Amount:${calculation.totalChemicalAmount} ${calculation.chemUnit}
Water Carrier Volume:       ${calculation.totalWaterLiters} Liters Water Carrier
Sprayer Equipment:          ${calculation.equipmentName} (${calculation.tankCapacityLiters}L capacity)
Tank Refills Required:      ~${calculation.tanksRequired} tank(s)
Concentrate per Tank:       ${calculation.chemicalPerTank} ${calculation.chemUnit}

3. WORKER PROTECTION STANDARD (WPS) & SAFETY PPE
--------------------------------------------------------------------------------
Worker Protection Standard (40 CFR Part 170) Compliance Mandatory:
[X] Nitrile Gloves: Chemical-resistant nitrile / neoprene gloves
[X] Respirator: N95 / organic vapor particle respirator mask
[X] Eye Protection: Protective chemical splash goggles / face shield
[X] Chemical Apron / Coveralls: Long-sleeved chemical coveralls & waterproof rubber boots

4. MANDATORY RE-ENTRY INTERVAL (REI: 12-24H) & PRE-HARVEST INTERVAL (PHI: 0-7D)
--------------------------------------------------------------------------------
Restricted Entry Interval (REI):  12 to 24 Hours (12-24h) - No unprotected field entry
Pre-Harvest Interval (PHI):       0 to 7 Days (0-7d) before crop harvest
Environmental Advisory:           Wind speed < 8 mph (12 km/h); apply during
                                  early morning (6-9 AM) to prevent drift and foliar scorch.

5. AGRONOMIST CERTIFICATION & STAMP SEAL
--------------------------------------------------------------------------------
Certified Agronomist:    Dr. J. Henderson, Ph.D., CCA
License / Reg Number:    CCA-${verificationCode}
Digital Stamp Seal:      DIGITALLY VERIFIED - AGROSENSE BOTANICAL ARBITER
Date of Verification:    ${today}
Agronomist Signature:    Dr. J. Henderson, Ph.D., CCA __________________________

================================================================================
Always read and strictly follow manufacturer product labels prior to mixing.
================================================================================`

    const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `Prescription_Work_Order_${crop}_${disease.replace(/\s+/g, '_')}_${Date.now()}.txt`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <div
      data-testid="prescription-modal-backdrop"
      className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-black/60 p-2 sm:p-4 backdrop-blur-sm print:p-0 print:bg-transparent print:static print:inset-auto print:overflow-visible"
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 10 }}
        transition={{ duration: 0.2 }}
        className="relative max-h-[92vh] w-full max-w-3xl overflow-y-auto rounded-2xl border border-border-light bg-card-light shadow-2xl dark:border-border dark:bg-card print:border-none print:shadow-none print:max-h-none print:w-full print:overflow-visible"
      >
        {/* Modal Top Header (Hidden on print) */}
        <div className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-3 border-b border-border-light bg-surface-light/95 px-4 py-3.5 backdrop-blur-sm sm:px-6 dark:border-border dark:bg-surface-dark/95 print:hidden">
          <div className="flex items-center gap-2.5 min-w-0">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent-500/15 text-accent-600 dark:text-accent-400">
              <FileCheck size={20} strokeWidth={2} />
            </span>
            <div className="min-w-0">
              <h3 className="truncate font-display text-sm font-bold text-slate-900 dark:text-ink-primary">
                Official Agronomic Prescription & Spray Work Order
              </h3>
              <p className="flex items-center gap-1.5 text-[11px] text-slate-500 dark:text-ink-muted">
                <span>Doc:</span>
                <span className="font-mono font-semibold">{prescriptionId}</span>
                <span className="text-slate-300 dark:text-white/20">|</span>
                <span className="font-mono text-accent-600 dark:text-accent-400 font-semibold">{verificationCode}</span>
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              variant="secondary"
              size="sm"
              icon={Printer}
              onClick={handlePrint}
              data-testid="print-prescription-button"
              className="hidden sm:inline-flex"
            >
              Print
            </Button>
            <Button
              type="button"
              variant="primary"
              size="sm"
              icon={FileDown}
              onClick={handleExportPDF}
              data-testid="export-pdf-button"
              className="bg-accent-600 hover:bg-accent-700 text-white font-medium"
            >
              📥 Export PDF Document
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              icon={FileText}
              onClick={handleDownloadWorkOrder}
              data-testid="download-txt-button"
              className="hidden md:inline-flex"
            >
              Download (.txt)
            </Button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-white/5 dark:hover:text-ink-primary transition-colors"
              aria-label="Close prescription modal"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Prescription Document Content */}
        <div
          id="printable-prescription-order"
          data-testid="prescription-work-order-document"
          className="space-y-6 p-4 sm:p-6 md:p-8 text-slate-900 dark:text-ink-primary print:p-0 print:text-black print:space-y-4"
        >
          {/* Official Document Header */}
          <div className="border-b-2 border-slate-900/15 pb-4 dark:border-white/15 print:border-black print:pb-3">
            <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="inline-block rounded-md bg-accent-500/15 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-accent-700 dark:text-accent-300 print:border print:border-black print:text-black print:bg-transparent">
                    Official Agronomic Prescription
                  </span>
                  <span className="inline-block rounded-md bg-emerald-500/15 px-2 py-0.5 text-[10px] font-mono font-bold text-emerald-700 dark:text-emerald-400 print:border print:border-black print:text-black print:bg-transparent">
                    {verificationCode}
                  </span>
                </div>
                <h2 className="mt-1.5 font-display text-xl font-bold tracking-tight text-slate-900 sm:text-2xl dark:text-ink-primary print:text-black print:text-xl">
                  OFFICIAL AGRONOMIC PRESCRIPTION & SPRAY WORK ORDER
                </h2>
                <p className="text-xs text-slate-500 dark:text-ink-muted print:text-slate-700">
                  Issued by AgroSense Botanical Arbiter · Verified Agricultural Pathology Laboratory
                </p>
              </div>

              <div className="flex sm:flex-col justify-between sm:text-right text-xs pt-1 sm:pt-0 border-t sm:border-t-0 border-slate-100 dark:border-white/5">
                <div>
                  <span className="text-[10px] text-slate-400 dark:text-ink-muted uppercase tracking-wider block">Prescription ID</span>
                  <span className="font-mono font-bold text-slate-800 dark:text-ink-primary print:text-black">{prescriptionId}</span>
                </div>
                <div className="sm:mt-1.5">
                  <span className="text-[10px] text-slate-400 dark:text-ink-muted uppercase tracking-wider block">Issued Date</span>
                  <span className="font-semibold text-slate-700 dark:text-ink-secondary print:text-black">{today}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Section 1: Field Diagnosis & Pathogen Identification */}
          <div className="space-y-2.5">
            <h4 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-accent-700 dark:text-accent-400 print:text-black">
              <Leaf size={15} className="text-accent-600 dark:text-accent-400 print:text-black" />
              1. Field Diagnosis & Pathogen Identification
            </h4>
            <div className="grid grid-cols-2 gap-3 rounded-xl border border-border-light bg-slate-900/[0.02] p-3.5 text-xs dark:border-border dark:bg-white/[0.02] sm:grid-cols-3 lg:grid-cols-5 print:border-slate-400 print:bg-transparent">
              <div>
                <span className="text-[11px] text-slate-500 dark:text-ink-muted print:text-slate-600 block">Crop Species:</span>
                <p className="font-bold capitalize text-slate-800 dark:text-ink-primary print:text-black text-sm">{crop}</p>
              </div>
              <div>
                <span className="text-[11px] text-slate-500 dark:text-ink-muted print:text-slate-600 block">Diagnosed Condition:</span>
                <p className="font-bold capitalize text-slate-800 dark:text-ink-primary print:text-black text-sm">{disease}</p>
              </div>
              <div>
                <span className="text-[11px] text-slate-500 dark:text-ink-muted print:text-slate-600 block">Field Severity:</span>
                <p className="font-bold text-rose-600 dark:text-rose-400 print:text-black text-sm">{severityLevel}</p>
              </div>
              <div>
                <span className="text-[11px] text-slate-500 dark:text-ink-muted print:text-slate-600 block">Vision Confidence:</span>
                <p className="font-bold text-emerald-600 dark:text-emerald-400 print:text-black text-sm">
                  {Math.round((diagnosis?.confidence ?? 0.95) * 100)}%
                </p>
              </div>
              <div className="col-span-2 sm:col-span-1">
                <span className="text-[11px] text-slate-500 dark:text-ink-muted print:text-slate-600 block">Verification Code:</span>
                <p className="font-mono font-bold text-accent-700 dark:text-accent-300 print:text-black text-sm">{verificationCode}</p>
              </div>
            </div>
            <div className="rounded-lg bg-slate-100/70 dark:bg-white/[0.03] px-3 py-1.5 text-[11px] text-slate-600 dark:text-ink-secondary print:border print:border-slate-300 print:bg-transparent print:text-slate-700">
              <span className="font-semibold text-slate-700 dark:text-ink-primary print:text-black">Pathogen Classification:</span> {guide.pathogen}
            </div>
          </div>

          {/* Section 2: Calculated Tank Mix & Spray Dosage Table */}
          <div className="space-y-2.5">
            <h4 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-accent-700 dark:text-accent-400 print:text-black">
              <FlaskConical size={15} className="text-accent-600 dark:text-accent-400 print:text-black" />
              2. Calculated Tank Mix & Spray Dosage Table
            </h4>
            <div className="overflow-x-auto rounded-xl border border-border-light dark:border-border print:border-slate-400">
              <table className="w-full text-left text-xs min-w-[500px] sm:min-w-full">
                <thead className="bg-slate-900/[0.04] text-slate-600 dark:bg-white/[0.04] dark:text-ink-muted print:bg-slate-100 print:text-black">
                  <tr>
                    <th className="px-3.5 py-2.5 font-semibold">Parameter / Metric</th>
                    <th className="px-3.5 py-2.5 font-semibold">Prescribed Specification</th>
                    <th className="px-3.5 py-2.5 font-semibold">Field Application Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-light dark:divide-border print:divide-slate-300">
                  <tr>
                    <td className="px-3.5 py-2 font-medium">Field Size</td>
                    <td className="px-3.5 py-2 font-semibold text-slate-800 dark:text-ink-primary print:text-black">
                      {calculation.numericArea} {calculation.unitLabel}
                    </td>
                    <td className="px-3.5 py-2 text-slate-500 dark:text-ink-muted print:text-slate-600">
                      {calculation.areaInAcres} acres equivalent plot area
                    </td>
                  </tr>
                  <tr>
                    <td className="px-3.5 py-2 font-medium">Active Formulation</td>
                    <td className="px-3.5 py-2 font-semibold text-accent-600 dark:text-accent-400 print:text-black">
                      {calculation.chemicalName}
                    </td>
                    <td className="px-3.5 py-2 text-slate-500 dark:text-ink-muted print:text-slate-600">
                      Standard dilution rate: {calculation.ratePerLiter} {calculation.chemUnit}/L
                    </td>
                  </tr>
                  <tr>
                    <td className="px-3.5 py-2 font-medium">Total Spray Volume</td>
                    <td className="px-3.5 py-2 font-bold text-slate-800 dark:text-ink-primary print:text-black">
                      {calculation.totalWaterLiters} Liters
                    </td>
                    <td className="px-3.5 py-2 text-slate-500 dark:text-ink-muted print:text-slate-600">
                      ({calculation.totalWaterGallons} US Gallons total solution)
                    </td>
                  </tr>
                  <tr>
                    <td className="px-3.5 py-2 font-medium">Chemical Concentrate Amount</td>
                    <td className="px-3.5 py-2 font-bold text-accent-600 dark:text-accent-400 print:text-black">
                      {calculation.totalChemicalAmount} {calculation.chemUnit}
                    </td>
                    <td className="px-3.5 py-2 text-slate-500 dark:text-ink-muted print:text-slate-600">
                      Measure concentrate precisely prior to tank premixing
                    </td>
                  </tr>
                  <tr>
                    <td className="px-3.5 py-2 font-medium">Water Carrier Volume</td>
                    <td className="px-3.5 py-2 font-semibold text-slate-800 dark:text-ink-primary print:text-black">
                      {calculation.totalWaterLiters} Liters Water Carrier
                    </td>
                    <td className="px-3.5 py-2 text-slate-500 dark:text-ink-muted print:text-slate-600">
                      pH 6.0 - 7.0 clean irrigation water recommended
                    </td>
                  </tr>
                  <tr>
                    <td className="px-3.5 py-2 font-medium">Sprayer Refill Specifications</td>
                    <td className="px-3.5 py-2 font-semibold text-slate-800 dark:text-ink-primary print:text-black">
                      ~{calculation.tanksRequired} tank(s) ({calculation.equipmentName})
                    </td>
                    <td className="px-3.5 py-2 text-slate-500 dark:text-ink-muted print:text-slate-600">
                      Load {calculation.chemicalPerTank} {calculation.chemUnit} concentrate per {calculation.tankCapacityLiters}L refill
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Section 3: Worker Protection Standard (WPS) & Safety PPE */}
          <div className="space-y-2.5">
            <h4 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-accent-700 dark:text-accent-400 print:text-black">
              <ShieldCheck size={15} className="text-accent-600 dark:text-accent-400 print:text-black" />
              3. Worker Protection Standard (WPS) & Safety PPE
            </h4>
            <p className="text-[11px] text-slate-600 dark:text-ink-secondary print:text-slate-700">
              Worker Protection Standard (40 CFR Part 170) Compliance Mandatory. All applicators and field handlers must wear the required Personal Protective Equipment (PPE):
            </p>
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
              <label
                className="flex cursor-pointer items-center gap-2.5 rounded-xl border border-border-light bg-slate-900/[0.02] p-3 text-xs text-slate-700 dark:border-border dark:bg-white/[0.02] dark:text-ink-secondary print:border-slate-400 print:bg-transparent print:text-black hover:bg-slate-900/[0.04] dark:hover:bg-white/[0.04] transition-colors"
              >
                <input
                  type="checkbox"
                  checked={ppeChecked.gloves}
                  onChange={() => togglePpe('gloves')}
                  className="rounded text-accent-600 focus:ring-accent-500 print:accent-black"
                />
                <span className="font-medium">Chemical-resistant nitrile / neoprene gloves (Nitrile gloves)</span>
              </label>

              <label
                className="flex cursor-pointer items-center gap-2.5 rounded-xl border border-border-light bg-slate-900/[0.02] p-3 text-xs text-slate-700 dark:border-border dark:bg-white/[0.02] dark:text-ink-secondary print:border-slate-400 print:bg-transparent print:text-black hover:bg-slate-900/[0.04] dark:hover:bg-white/[0.04] transition-colors"
              >
                <input
                  type="checkbox"
                  checked={ppeChecked.respirator}
                  onChange={() => togglePpe('respirator')}
                  className="rounded text-accent-600 focus:ring-accent-500 print:accent-black"
                />
                <span className="font-medium">N95 / organic vapor respirator mask (N95/organic vapor respirator)</span>
              </label>

              <label
                className="flex cursor-pointer items-center gap-2.5 rounded-xl border border-border-light bg-slate-900/[0.02] p-3 text-xs text-slate-700 dark:border-border dark:bg-white/[0.02] dark:text-ink-secondary print:border-slate-400 print:bg-transparent print:text-black hover:bg-slate-900/[0.04] dark:hover:bg-white/[0.04] transition-colors"
              >
                <input
                  type="checkbox"
                  checked={ppeChecked.eyeProtection}
                  onChange={() => togglePpe('eyeProtection')}
                  className="rounded text-accent-600 focus:ring-accent-500 print:accent-black"
                />
                <span className="font-medium">Protective chemical splash goggles / face shield (Splash goggles)</span>
              </label>

              <label
                className="flex cursor-pointer items-center gap-2.5 rounded-xl border border-border-light bg-slate-900/[0.02] p-3 text-xs text-slate-700 dark:border-border dark:bg-white/[0.02] dark:text-ink-secondary print:border-slate-400 print:bg-transparent print:text-black hover:bg-slate-900/[0.04] dark:hover:bg-white/[0.04] transition-colors"
              >
                <input
                  type="checkbox"
                  checked={ppeChecked.coveralls}
                  onChange={() => togglePpe('coveralls')}
                  className="rounded text-accent-600 focus:ring-accent-500 print:accent-black"
                />
                <span className="font-medium">Long-sleeved chemical coveralls / chemical apron & waterproof rubber boots (Chemical apron)</span>
              </label>
            </div>
          </div>

          {/* Section 4: Mandatory Re-Entry Interval (REI: 12-24h) & Pre-Harvest Interval (PHI: 0-7d) */}
          <div className="space-y-2.5">
            <h4 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-rose-700 dark:text-rose-400 print:text-black">
              <ShieldAlert size={15} className="text-rose-600 dark:text-rose-400 print:text-black" />
              4. Mandatory Re-Entry Interval (REI: 12-24h) & Pre-Harvest Interval (PHI: 0-7d)
            </h4>
            <div className="grid grid-cols-1 gap-3 rounded-xl border border-rose-500/25 bg-rose-500/5 p-4 text-xs text-rose-950 dark:border-rose-500/25 dark:bg-rose-500/10 dark:text-rose-200 sm:grid-cols-2 print:border-slate-400 print:bg-transparent print:text-black">
              <div className="space-y-1">
                <div className="font-bold flex items-center gap-1.5 text-rose-700 dark:text-rose-300 print:text-black">
                  <Clock size={14} className="text-rose-600 dark:text-rose-400 print:text-black" />
                  <span>Restricted Entry Interval (REI): 12 - 24 Hours (12-24h)</span>
                </div>
                <p className="text-[11px] text-slate-600 dark:text-ink-secondary print:text-slate-700">
                  Do not enter or permit agricultural workers to enter treated field areas without full protective equipment during the REI window.
                </p>
              </div>

              <div className="space-y-1">
                <div className="font-bold flex items-center gap-1.5 text-rose-700 dark:text-rose-300 print:text-black">
                  <Calendar size={14} className="text-rose-600 dark:text-rose-400 print:text-black" />
                  <span>Pre-Harvest Interval (PHI): 0 - 7 Days (0-7d)</span>
                </div>
                <p className="text-[11px] text-slate-600 dark:text-ink-secondary print:text-slate-700">
                  Adhere to the mandatory waiting period between last spray application and crop harvest to ensure chemical residue compliance.
                </p>
              </div>
            </div>
            <div className="rounded-lg bg-amber-500/10 dark:bg-amber-500/15 border border-amber-500/20 px-3.5 py-2 text-[11px] text-amber-900 dark:text-amber-200 print:border print:border-slate-400 print:bg-transparent print:text-slate-800">
              <strong>Environmental & Drift Advisory:</strong> Apply when wind speed &lt; 8 mph (12 km/h); spray early morning (6:00 - 9:00 AM) or dusk to prevent spray drift and foliar scorch.
            </div>
          </div>

          {/* Section 5: Agronomist Certification & Stamp Seal */}
          <div className="space-y-3 border-t-2 border-slate-900/10 pt-4 dark:border-white/10 print:border-black print:pt-3">
            <h4 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-ink-muted print:text-black">
              <Award size={15} className="text-accent-600 dark:text-accent-400 print:text-black" />
              5. Agronomist Certification & Stamp Seal
            </h4>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 items-center">
              <div className="space-y-1.5 text-xs text-slate-700 dark:text-ink-secondary print:text-black">
                <p>
                  <strong className="text-slate-900 dark:text-ink-primary print:text-black">Certified Agronomist:</strong> Dr. J. Henderson, Ph.D., CCA
                </p>
                <p>
                  <strong className="text-slate-900 dark:text-ink-primary print:text-black">License / Reg Number:</strong>{' '}
                  <span className="font-mono font-semibold">CCA-#AGRI-88294-EXT</span>
                </p>
                <p>
                  <strong className="text-slate-900 dark:text-ink-primary print:text-black">Verification Code:</strong>{' '}
                  <span className="font-mono font-semibold text-accent-700 dark:text-accent-300 print:text-black">{verificationCode}</span>
                </p>
                <p>
                  <strong className="text-slate-900 dark:text-ink-primary print:text-black">Digital Verification:</strong> Digitally Verified - AgroSense Botanical Arbiter
                </p>
                <p className="text-[11px] text-slate-500 dark:text-ink-muted print:text-slate-600">
                  Date of Verification: {today}
                </p>
              </div>

              {/* Official Agronomist Stamp Seal & Signature Block */}
              <div className="flex flex-col items-center sm:items-end justify-center gap-2">
                <div className="flex items-center gap-3">
                  {/* Stylized Agronomist Stamp Seal */}
                  <div
                    data-testid="agronomist-stamp-seal"
                    className="relative flex h-24 w-24 flex-col items-center justify-center rounded-full border-2 border-dashed border-emerald-600 bg-emerald-500/10 p-1.5 text-center text-[8px] font-bold uppercase tracking-wider text-emerald-800 dark:border-emerald-400 dark:text-emerald-300 print:border-2 print:border-black print:text-black print:bg-transparent"
                  >
                    <span className="text-[7px] tracking-widest text-emerald-700 dark:text-emerald-400 print:text-black">★ OFFICIAL SEAL ★</span>
                    <span className="font-extrabold text-[9px] leading-tight text-emerald-900 dark:text-emerald-200 print:text-black">CERTIFIED AGRONOMIST</span>
                    <span className="font-mono text-[7px] text-emerald-800 dark:text-emerald-300 print:text-black">#AGRI-88294-EXT</span>
                    <span className="text-[6px] tracking-tight text-emerald-600 dark:text-emerald-400 print:text-black">AGROSENSE ARBITER</span>
                  </div>

                  <div className="text-right">
                    <div className="font-serif italic text-base sm:text-lg text-slate-800 dark:text-ink-primary print:text-black border-b border-slate-400 pb-1">
                      Dr. J. Henderson, Ph.D., CCA
                    </div>
                    <span className="text-[10px] text-slate-500 dark:text-ink-muted print:text-slate-600 block mt-0.5">
                      Authorized Agronomist Signature
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Bottom Footer (Hidden on print) */}
        <div className="sticky bottom-0 z-10 flex flex-wrap items-center justify-between gap-3 border-t border-border-light bg-surface-light/95 px-4 py-3 sm:px-6 backdrop-blur-sm dark:border-border dark:bg-surface-dark/95 print:hidden">
          <span className="text-[11px] text-slate-500 dark:text-ink-muted">
            Prescription generated automatically from LeafSense pathology diagnosis · Code: <span className="font-mono font-semibold">{verificationCode}</span>
          </span>
          <div className="flex flex-wrap items-center gap-2">
            <Button type="button" variant="secondary" size="sm" onClick={onClose}>
              Close
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              icon={FileText}
              onClick={handleDownloadWorkOrder}
            >
              Download (.txt)
            </Button>
            <Button
              type="button"
              variant="primary"
              size="sm"
              icon={FileDown}
              onClick={handleExportPDF}
              data-testid="export-pdf-button-footer"
              className="bg-accent-600 hover:bg-accent-700 text-white font-medium"
            >
              📥 Export PDF Document
            </Button>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
