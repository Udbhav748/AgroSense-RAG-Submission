import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  AlertCircle,
  Calculator,
  Check,
  Droplets,
  FileDown,
  FileText,
  FlaskConical,
  Gauge,
  HelpCircle,
  Info,
  Layers,
  RotateCcw,
  Scale,
  Shield,
  Wind,
} from 'lucide-react'
import Button from '../ui/Button'
import PrescriptionWorkOrderModal from './PrescriptionWorkOrderModal'
import {
  AREA_UNITS,
  CHEMICAL_PRESETS,
  SPRAY_EQUIPMENT_PRESETS,
  calculateSprayDosage,
} from '../../utils/agronomyData'

/**
 * Interactive Spray Dosage & Mix Volume Calculator Widget.
 * Calculates required water mix volume and chemical concentrates for any field size.
 */
export default function SprayDosageCalculator({ defaultDisease = '', defaultCrop = '' }) {
  const [areaValue, setAreaValue] = useState('1000')
  const [unitId, setUnitId] = useState('sq_ft')
  const [chemicalPresetId, setChemicalPresetId] = useState('copper')
  const [customRate, setCustomRate] = useState('2.0')
  const [customUnit, setCustomUnit] = useState('g')
  const [equipmentId, setEquipmentId] = useState('knapsack')
  const [showPrescriptionModal, setShowPrescriptionModal] = useState(false)

  // Calculate results on the fly
  const results = useMemo(() => {
    return calculateSprayDosage({
      areaValue,
      unitId,
      chemicalPresetId,
      customRatePerLiter: customRate,
      customUnit,
      equipmentId,
    })
  }, [areaValue, unitId, chemicalPresetId, customRate, customUnit, equipmentId])

  const handleResetDefaults = () => {
    setAreaValue('1000')
    setUnitId('sq_ft')
    setChemicalPresetId('copper')
    setEquipmentId('knapsack')
    setCustomRate('2.0')
    setCustomUnit('g')
  }

  return (
    <div className="panel rounded-panel border border-border-light p-5 shadow-soft dark:border-border">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border-light pb-4 dark:border-border">
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-500/15 text-accent-600 dark:text-accent-400">
            <Calculator size={16} strokeWidth={2} />
          </span>
          <div>
            <h3 className="font-display text-base font-bold text-slate-900 dark:text-ink-primary">
              Field Spray Dosage & Tank Mix Calculator
            </h3>
            <p className="text-xs text-slate-500 dark:text-ink-muted">
              Compute water volumes, chemical amounts, and tank refills tailored to your plot size.
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={handleResetDefaults}
          className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-600 dark:hover:text-ink-secondary"
          title="Reset to defaults"
        >
          <RotateCcw size={12} />
          <span>Reset</span>
        </button>
      </div>

      {/* Grid: Inputs (Left) and Results (Right) */}
      <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Left Column: Form Controls */}
        <div className="space-y-4 lg:col-span-6">
          {/* 1. Field Size & Unit */}
          <div className="space-y-1.5">
            <label
              htmlFor="field-size-input"
              className="flex items-center justify-between text-xs font-semibold text-slate-700 dark:text-ink-secondary"
            >
              <span>Field / Treatment Area</span>
              <span className="text-[11px] font-normal text-slate-400">
                {results.areaInAcres} acres equivalent
              </span>
            </label>

            <div className="flex gap-2">
              <div className="relative flex-1">
                <input
                  id="field-size-input"
                  data-testid="field-size-input"
                  type="number"
                  min="0"
                  step="any"
                  value={areaValue}
                  onChange={(e) => setAreaValue(e.target.value)}
                  placeholder="e.g. 500 or 2"
                  className="w-full rounded-lg border border-border-light bg-white/70 px-3 py-2 text-sm font-semibold text-slate-800 outline-none transition-colors focus:border-accent-500 focus:ring-2 focus:ring-accent-500/20 dark:border-border dark:bg-white/[0.03] dark:text-ink-primary"
                />
              </div>

              <select
                id="area-unit-select"
                data-testid="area-unit-select"
                aria-label="Area unit"
                value={unitId}
                onChange={(e) => setUnitId(e.target.value)}
                className="rounded-lg border border-border-light bg-white/70 px-3 py-2 text-xs font-medium text-slate-700 outline-none transition-colors focus:border-accent-500 focus:ring-2 focus:ring-accent-500/20 dark:border-border dark:bg-surface-dark-secondary dark:text-ink-primary"
              >
                {AREA_UNITS.map((u) => (
                  <option key={u.id} value={u.id}>
                    {u.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* 2. Treatment Chemical Selection */}
          <div className="space-y-1.5">
            <label
              htmlFor="chemical-preset-select"
              className="flex items-center justify-between text-xs font-semibold text-slate-700 dark:text-ink-secondary"
            >
              <span>Treatment Active Formulation</span>
              <span className="text-[11px] font-normal text-accent-600 dark:text-accent-400">
                {results.ratePerLiter} {results.chemUnit}/L water
              </span>
            </label>

            <select
              id="chemical-preset-select"
              data-testid="chemical-preset-select"
              value={chemicalPresetId}
              onChange={(e) => setChemicalPresetId(e.target.value)}
              className="w-full rounded-lg border border-border-light bg-white/70 px-3 py-2 text-xs font-medium text-slate-700 outline-none transition-colors focus:border-accent-500 focus:ring-2 focus:ring-accent-500/20 dark:border-border dark:bg-surface-dark-secondary dark:text-ink-primary"
            >
              {CHEMICAL_PRESETS.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.dosagePerLiter} {c.unit}/L)
                </option>
              ))}
            </select>
          </div>

          {/* Custom Chemical Rate Inputs if custom selected */}
          {chemicalPresetId === 'custom' && (
            <div className="flex gap-2 rounded-lg border border-border-light bg-slate-900/[0.02] p-2.5 dark:border-border dark:bg-white/[0.02]">
              <div className="flex-1">
                <label className="text-[10px] font-medium text-slate-500 dark:text-ink-muted">
                  Custom Dosage Rate
                </label>
                <input
                  type="number"
                  min="0"
                  step="any"
                  value={customRate}
                  onChange={(e) => setCustomRate(e.target.value)}
                  placeholder="Rate per liter"
                  className="mt-1 w-full rounded border border-border-light bg-white px-2 py-1 text-xs text-slate-800 dark:border-border dark:bg-white/[0.04] dark:text-ink-primary"
                />
              </div>
              <div className="w-24">
                <label className="text-[10px] font-medium text-slate-500 dark:text-ink-muted">
                  Unit
                </label>
                <select
                  value={customUnit}
                  onChange={(e) => setCustomUnit(e.target.value)}
                  className="mt-1 w-full rounded border border-border-light bg-white px-2 py-1 text-xs text-slate-800 dark:border-border dark:bg-surface-dark-secondary dark:text-ink-primary"
                >
                  <option value="g">grams (g)</option>
                  <option value="ml">ml (liquid)</option>
                  <option value="kg">kg</option>
                  <option value="oz">oz</option>
                </select>
              </div>
            </div>
          )}

          {/* 3. Sprayer Equipment */}
          <div className="space-y-1.5">
            <label
              htmlFor="equipment-preset-select"
              className="flex items-center justify-between text-xs font-semibold text-slate-700 dark:text-ink-secondary"
            >
              <span>Sprayer Equipment & Tank Size</span>
              <span className="text-[11px] font-normal text-slate-400">
                {results.tankCapacityLiters}L capacity
              </span>
            </label>

            <select
              id="equipment-preset-select"
              data-testid="equipment-preset-select"
              value={equipmentId}
              onChange={(e) => setEquipmentId(e.target.value)}
              className="w-full rounded-lg border border-border-light bg-white/70 px-3 py-2 text-xs font-medium text-slate-700 outline-none transition-colors focus:border-accent-500 focus:ring-2 focus:ring-accent-500/20 dark:border-border dark:bg-surface-dark-secondary dark:text-ink-primary"
            >
              {SPRAY_EQUIPMENT_PRESETS.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Right Column: Calculated Mix Breakdown Cards */}
        <div className="space-y-3 lg:col-span-6">
          <div className="rounded-xl border border-accent-500/20 bg-accent-500/5 p-4 dark:border-accent-500/20 dark:bg-accent-500/10">
            <div className="flex items-center justify-between border-b border-accent-500/20 pb-3">
              <span className="text-xs font-bold uppercase tracking-wider text-accent-700 dark:text-accent-300">
                Calculated Tank Mix Prescription
              </span>
              <span className="rounded bg-accent-500/20 px-2 py-0.5 text-[10px] font-bold text-accent-800 dark:text-accent-200">
                {results.numericArea} {results.unitLabel}
              </span>
            </div>

            {/* Key Output Stats */}
            <div className="mt-3 grid grid-cols-2 gap-3">
              {/* Total Water Mix */}
              <div className="rounded-lg bg-white/80 p-3 shadow-sm dark:bg-slate-900/60">
                <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-ink-muted">
                  <Droplets size={13} className="text-blue-500" />
                  <span>Total Water Mix</span>
                </div>
                <div data-testid="water-volume-liters" className="mt-1 font-display text-xl font-bold text-slate-900 dark:text-ink-primary">
                  {results.totalWaterLiters} L
                </div>
                <span className="text-[10px] text-slate-400 dark:text-ink-muted">
                  ({results.totalWaterGallons} US Gallons)
                </span>
              </div>

              {/* Required Chemical Concentrate */}
              <div className="rounded-lg bg-white/80 p-3 shadow-sm dark:bg-slate-900/60">
                <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-ink-muted">
                  <FlaskConical size={13} className="text-accent-500" />
                  <span>Required Chemical</span>
                </div>
                <div data-testid="chemical-amount-output" className="mt-1 font-display text-xl font-bold text-accent-600 dark:text-accent-400">
                  {results.totalChemicalAmount} {results.chemUnit}
                </div>
                <span className="text-[10px] text-slate-400 dark:text-ink-muted">
                  @ {results.ratePerLiter} {results.chemUnit}/L dilution
                </span>
              </div>
            </div>

            {/* Sprayer Refill Guide */}
            <div className="mt-3 rounded-lg border border-accent-500/20 bg-white/60 p-3 text-xs text-slate-700 dark:bg-slate-900/40 dark:text-ink-secondary">
              <div className="flex items-center justify-between font-semibold text-slate-900 dark:text-ink-primary">
                <div className="flex items-center gap-1.5">
                  <Layers size={13} className="text-accent-500" />
                  <span>Equipment Tank Refills</span>
                </div>
                <span data-testid="tank-refills-count" className="font-bold text-accent-600 dark:text-accent-400">
                  ~{results.tanksRequired} tank{results.tanksRequired === 1 ? '' : 's'}
                </span>
              </div>
              <p className="mt-1 text-[11px] text-slate-500 dark:text-ink-muted">
                Add <strong className="font-semibold text-slate-800 dark:text-ink-primary">{results.chemicalPerTank} {results.chemUnit}</strong> of {results.chemicalName} per full {results.tankCapacityLiters}L tank fill.
              </p>
            </div>

            {/* Action button to generate prescription work order */}
            <div className="pt-2">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                icon={FileText}
                onClick={() => setShowPrescriptionModal(true)}
                className="w-full justify-center"
              >
                Generate Spray Work Order Document
              </Button>
            </div>
          </div>

          {/* Meteorological & Application Advisory */}
          <div className="rounded-xl border border-border-light bg-slate-900/[0.02] p-3 text-xs text-slate-600 dark:border-border dark:bg-white/[0.02] dark:text-ink-secondary">
            <div className="mb-1 flex items-center gap-1.5 font-semibold text-slate-800 dark:text-ink-primary">
              <Wind size={13} className="text-accent-500" />
              <span>Optimal Spraying Conditions</span>
            </div>
            <ul className="space-y-1 text-[11px] text-slate-500 dark:text-ink-muted">
              <li>• Spray during calm conditions (wind speed &lt; 8 mph / 12 km/h) to prevent drift.</li>
              <li>• Apply at early morning (6-9 AM) or dusk to avoid rapid evaporation and scorch.</li>
              <li>• Always calibrate nozzle pressure for even foliage canopy droplet distribution.</li>
            </ul>
          </div>
        </div>
      </div>

      <PrescriptionWorkOrderModal
        isOpen={showPrescriptionModal}
        onClose={() => setShowPrescriptionModal(false)}
        diagnosis={{
          crop: defaultCrop || 'Field Crop',
          disease: defaultDisease || 'Foliar Treatment',
          confidence: 0.95,
        }}
        severity={{ level: 'Moderate', description: 'Preventative / targeted intervention' }}
        dosageInfo={results}
      />
    </div>
  )
}
