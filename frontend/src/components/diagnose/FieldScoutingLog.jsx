import { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  Calendar,
  CheckCircle2,
  Clock,
  Download,
  FileJson,
  FileSpreadsheet,
  Filter,
  Layers,
  Leaf,
  MapPin,
  Plus,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Sprout,
  Trash2,
  X,
} from 'lucide-react'
import Button from '../ui/Button'
import {
  getScoutingLogs,
  saveScoutingLogs,
  addScoutingEntry,
  deleteScoutingEntry,
  resetScoutingLogs,
  exportScoutingToCSV,
  exportScoutingToJSON,
  SCOUTING_EVENT_NAME,
} from '../../utils/scoutingStorage'

function getSeverityBadgeStyle(severity) {
  const sev = (severity || '').toLowerCase()
  if (sev === 'severe' || sev === 'high' || sev === 'critical') {
    return 'bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-500/30'
  }
  if (sev === 'moderate' || sev === 'medium') {
    return 'bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30'
  }
  return 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30'
}

function formatDate(isoString) {
  if (!isoString) return 'N/A'
  try {
    const d = new Date(isoString)
    if (isNaN(d.getTime())) return isoString
    return d.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return isoString
  }
}

/**
 * Field Scouting Log & Outbreak History Component
 * Allows farmers to view, filter, add, and export historical crop disease findings.
 */
export default function FieldScoutingLog({ currentDiagnosis = null, onEntryAdded = null }) {
  const [logs, setLogs] = useState(() => getScoutingLogs())
  const [selectedCrop, setSelectedCrop] = useState('ALL')
  const [selectedSeverity, setSelectedSeverity] = useState('ALL')
  const [searchQuery, setSearchQuery] = useState('')
  const [showAddModal, setShowAddModal] = useState(false)
  const [feedbackMessage, setFeedbackMessage] = useState(null)

  // Form State for New Scouting Entry
  const [formCrop, setFormCrop] = useState(currentDiagnosis?.crop || 'Tomato')
  const [formDisease, setFormDisease] = useState(currentDiagnosis?.disease || '')
  const [formSeverity, setFormSeverity] = useState('Moderate')
  const [formConfidence, setFormConfidence] = useState(
    currentDiagnosis?.confidence != null ? Math.round(currentDiagnosis.confidence * 100) : 95
  )
  const [formLocation, setFormLocation] = useState('North Orchard Block B')
  const [formNotes, setFormNotes] = useState('')
  const [formRemedy, setFormRemedy] = useState('')

  // Sync with global storage and window events
  useEffect(() => {
    const handleStorageUpdate = (e) => {
      if (e?.detail) {
        setLogs(e.detail)
      } else {
        setLogs(getScoutingLogs())
      }
    }

    window.addEventListener(SCOUTING_EVENT_NAME, handleStorageUpdate)
    window.addEventListener('storage', handleStorageUpdate)

    return () => {
      window.removeEventListener(SCOUTING_EVENT_NAME, handleStorageUpdate)
      window.removeEventListener('storage', handleStorageUpdate)
    }
  }, [])

  // Sync form inputs when currentDiagnosis changes
  useEffect(() => {
    if (currentDiagnosis) {
      if (currentDiagnosis.crop) setFormCrop(currentDiagnosis.crop)
      if (currentDiagnosis.disease) setFormDisease(currentDiagnosis.disease)
      if (currentDiagnosis.confidence != null) {
        setFormConfidence(Math.round(currentDiagnosis.confidence * 100))
      }
    }
  }, [currentDiagnosis])

  // Unique Crops for Filter Dropdown
  const uniqueCrops = useMemo(() => {
    const cropSet = new Set(logs.map((log) => (log.crop ? log.crop.trim() : '').toLowerCase()))
    const crops = Array.from(cropSet).filter(Boolean)
    return crops.map((c) => c.charAt(0).toUpperCase() + c.slice(1))
  }, [logs])

  // Summary Metrics
  const summaryMetrics = useMemo(() => {
    const totalScouts = logs.length
    const highSeverityOutbreaks = logs.filter((l) => {
      const s = (l.severity || '').toLowerCase()
      return s === 'severe' || s === 'high' || s === 'critical'
    }).length

    // Most Prevalent Pathogen
    const diseaseCounts = {}
    logs.forEach((l) => {
      if (l.disease && !l.disease.toLowerCase().includes('healthy')) {
        const d = l.disease.trim()
        diseaseCounts[d] = (diseaseCounts[d] || 0) + 1
      }
    })

    let mostPrevalentPathogen = 'None Detected'
    let highestCount = 0
    Object.entries(diseaseCounts).forEach(([disease, count]) => {
      if (count > highestCount) {
        highestCount = count
        mostPrevalentPathogen = disease
      }
    })

    // Last Scout Date
    let lastScoutDate = 'No Entries'
    if (logs.length > 0) {
      const sorted = [...logs].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
      lastScoutDate = formatDate(sorted[0].timestamp)
    }

    return {
      totalScouts,
      highSeverityOutbreaks,
      mostPrevalentPathogen,
      lastScoutDate,
    }
  }, [logs])

  // Filtered Scouting Logs
  const filteredLogs = useMemo(() => {
    return logs.filter((log) => {
      // Filter Crop
      if (selectedCrop !== 'ALL') {
        if ((log.crop || '').toLowerCase() !== selectedCrop.toLowerCase()) {
          return false
        }
      }

      // Filter Severity
      if (selectedSeverity !== 'ALL') {
        if ((log.severity || '').toLowerCase() !== selectedSeverity.toLowerCase()) {
          return false
        }
      }

      // Search Query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase()
        const matchCrop = (log.crop || '').toLowerCase().includes(q)
        const matchDisease = (log.disease || '').toLowerCase().includes(q)
        const matchLocation = (log.location || '').toLowerCase().includes(q)
        const matchNotes = (log.notes || '').toLowerCase().includes(q)
        const matchRemedy = (log.remedyApplied || '').toLowerCase().includes(q)

        if (!matchCrop && !matchDisease && !matchLocation && !matchNotes && !matchRemedy) {
          return false
        }
      }

      return true
    })
  }, [logs, selectedCrop, selectedSeverity, searchQuery])

  // Quick Action: Save Current Diagnosis
  const handleLogCurrentDiagnosis = () => {
    if (!currentDiagnosis) return

    const newRecord = addScoutingEntry({
      crop: currentDiagnosis.crop || 'Crop',
      disease: currentDiagnosis.disease || 'Diagnosed Condition',
      severity: currentDiagnosis.severity || (currentDiagnosis.confidence > 0.8 ? 'Severe' : 'Moderate'),
      confidence: currentDiagnosis.confidence ?? 0.95,
      location: formLocation || 'Field Zone 1 (Current Scout)',
      notes: `Automated log from live LeafSense diagnosis (${currentDiagnosis.raw_class || 'vision'})`,
      remedyApplied: 'Standard Treatment Recommended',
    })

    setLogs(getScoutingLogs())
    setFeedbackMessage('Current diagnosis logged to scouting history!')
    setTimeout(() => setFeedbackMessage(null), 3500)
    onEntryAdded?.(newRecord)
  }

  // Handle Submit New Entry Form
  const handleCreateEntry = (e) => {
    e.preventDefault()
    if (!formCrop.trim() || !formDisease.trim() || !formLocation.trim()) {
      return
    }

    const newRecord = addScoutingEntry({
      crop: formCrop.trim(),
      disease: formDisease.trim(),
      severity: formSeverity,
      confidence: Number(formConfidence) / 100,
      location: formLocation.trim(),
      notes: formNotes.trim(),
      remedyApplied: formRemedy.trim() || 'Pending Assessment',
    })

    setLogs(getScoutingLogs())
    setShowAddModal(false)
    setFormNotes('')
    setFeedbackMessage('New scouting observation logged successfully!')
    setTimeout(() => setFeedbackMessage(null), 3500)
    onEntryAdded?.(newRecord)
  }

  const handleDelete = (id) => {
    const updated = deleteScoutingEntry(id)
    setLogs(updated)
  }

  const handleResetDefaults = () => {
    const defaults = resetScoutingLogs()
    setLogs(defaults)
    setFeedbackMessage('Scouting history reset to demonstration baseline.')
    setTimeout(() => setFeedbackMessage(null), 3000)
  }

  return (
    <div
      data-testid="field-scouting-log-container"
      className="space-y-6 rounded-2xl border border-border-light bg-card-light p-5 shadow-soft dark:border-border dark:bg-card"
    >
      {/* Header & Quick Actions */}
      <div className="flex flex-col gap-4 border-b border-border-light pb-5 sm:flex-row sm:items-center sm:justify-between dark:border-border">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-500/15 text-accent-600 dark:text-accent-400">
              <MapPin size={18} strokeWidth={2} />
            </span>
            <h2 className="font-display text-lg font-bold tracking-tight text-slate-900 dark:text-ink-primary">
              Field Scouting Log & Outbreak History
            </h2>
          </div>
          <p className="text-xs text-slate-500 dark:text-ink-muted">
            Track historical crop pathologies, field sector outbreaks, treatments applied, and severity trends.
          </p>
        </div>

        {/* Action Buttons: Log current, Add new, Export CSV, Export JSON */}
        <div className="flex flex-wrap items-center gap-2">
          {currentDiagnosis && (
            <Button
              type="button"
              variant="primary"
              size="sm"
              icon={Sparkles}
              onClick={handleLogCurrentDiagnosis}
              data-testid="log-current-diagnosis-button"
            >
              [ + Log Current Diagnosis to Scouting History ]
            </Button>
          )}

          <Button
            type="button"
            variant="secondary"
            size="sm"
            icon={Plus}
            onClick={() => setShowAddModal(true)}
            data-testid="open-add-scout-modal-button"
          >
            Add Scout Log
          </Button>

          <Button
            type="button"
            variant="ghost"
            size="sm"
            icon={FileSpreadsheet}
            onClick={() => exportScoutingToCSV(filteredLogs)}
            data-testid="export-csv-button"
            title="Export scouting records to CSV"
          >
            CSV
          </Button>

          <Button
            type="button"
            variant="ghost"
            size="sm"
            icon={FileJson}
            onClick={() => exportScoutingToJSON(filteredLogs)}
            data-testid="export-json-button"
            title="Export scouting records to JSON"
          >
            JSON
          </Button>
        </div>
      </div>

      {/* Feedback Toast Notification */}
      <AnimatePresence>
        {feedbackMessage && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            data-testid="scouting-feedback-message"
            className="flex items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-2.5 text-xs font-semibold text-emerald-800 dark:text-emerald-300"
          >
            <CheckCircle2 size={16} className="text-emerald-600 dark:text-emerald-400" />
            <span>{feedbackMessage}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Interactive Summary Metrics Cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4" data-testid="scouting-summary-cards">
        {/* Card 1: Total Scouts */}
        <div className="rounded-xl border border-border-light bg-slate-900/[0.02] p-3.5 dark:border-border dark:bg-white/[0.02]">
          <div className="flex items-center justify-between text-slate-500 dark:text-ink-muted">
            <span className="text-[11px] font-medium uppercase tracking-wider">Total Scouts</span>
            <Activity size={15} className="text-accent-500" />
          </div>
          <div data-testid="metric-total-scouts" className="mt-1 font-display text-2xl font-bold text-slate-900 dark:text-ink-primary">
            {summaryMetrics.totalScouts}
          </div>
          <span className="text-[10px] text-slate-400 dark:text-ink-muted">Logged field observations</span>
        </div>

        {/* Card 2: High Severity Outbreaks */}
        <div className="rounded-xl border border-border-light bg-slate-900/[0.02] p-3.5 dark:border-border dark:bg-white/[0.02]">
          <div className="flex items-center justify-between text-slate-500 dark:text-ink-muted">
            <span className="text-[11px] font-medium uppercase tracking-wider">High Severity</span>
            <ShieldAlert size={15} className="text-rose-500" />
          </div>
          <div data-testid="metric-high-severity" className="mt-1 font-display text-2xl font-bold text-rose-600 dark:text-rose-400">
            {summaryMetrics.highSeverityOutbreaks}
          </div>
          <span className="text-[10px] text-slate-400 dark:text-ink-muted">Severe & Critical threats</span>
        </div>

        {/* Card 3: Most Prevalent Pathogen */}
        <div className="rounded-xl border border-border-light bg-slate-900/[0.02] p-3.5 dark:border-border dark:bg-white/[0.02]">
          <div className="flex items-center justify-between text-slate-500 dark:text-ink-muted">
            <span className="text-[11px] font-medium uppercase tracking-wider">Top Pathogen</span>
            <Sprout size={15} className="text-amber-500" />
          </div>
          <div data-testid="metric-top-pathogen" className="mt-1 truncate font-display text-base font-bold text-slate-900 dark:text-ink-primary" title={summaryMetrics.mostPrevalentPathogen}>
            {summaryMetrics.mostPrevalentPathogen}
          </div>
          <span className="text-[10px] text-slate-400 dark:text-ink-muted">Highest recorded incidence</span>
        </div>

        {/* Card 4: Last Scout Date */}
        <div className="rounded-xl border border-border-light bg-slate-900/[0.02] p-3.5 dark:border-border dark:bg-white/[0.02]">
          <div className="flex items-center justify-between text-slate-500 dark:text-ink-muted">
            <span className="text-[11px] font-medium uppercase tracking-wider">Last Scout Date</span>
            <Calendar size={15} className="text-emerald-500" />
          </div>
          <div data-testid="metric-last-scout-date" className="mt-1 truncate font-display text-xs font-bold text-slate-900 dark:text-ink-primary" title={summaryMetrics.lastScoutDate}>
            {summaryMetrics.lastScoutDate}
          </div>
          <span className="text-[10px] text-slate-400 dark:text-ink-muted">Most recent field visit</span>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col gap-3 rounded-xl border border-border-light bg-slate-900/[0.02] p-3.5 dark:border-border dark:bg-white/[0.02] sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-3">
          {/* Crop Filter */}
          <div className="flex items-center gap-1.5">
            <Filter size={14} className="text-slate-400 dark:text-ink-muted" />
            <label htmlFor="scouting-crop-filter" className="text-xs font-semibold text-slate-600 dark:text-ink-secondary">
              Crop:
            </label>
            <select
              id="scouting-crop-filter"
              data-testid="filter-crop-select"
              value={selectedCrop}
              onChange={(e) => setSelectedCrop(e.target.value)}
              className="rounded-lg border border-border-light bg-surface-light px-2.5 py-1 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-border dark:bg-surface-dark dark:text-ink-primary"
            >
              <option value="ALL">All Crops ({logs.length})</option>
              {uniqueCrops.map((crop) => (
                <option key={crop} value={crop}>
                  {crop}
                </option>
              ))}
            </select>
          </div>

          {/* Severity Filter */}
          <div className="flex items-center gap-1.5">
            <label htmlFor="scouting-severity-filter" className="text-xs font-semibold text-slate-600 dark:text-ink-secondary">
              Severity:
            </label>
            <select
              id="scouting-severity-filter"
              data-testid="filter-severity-select"
              value={selectedSeverity}
              onChange={(e) => setSelectedSeverity(e.target.value)}
              className="rounded-lg border border-border-light bg-surface-light px-2.5 py-1 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-border dark:bg-surface-dark dark:text-ink-primary"
            >
              <option value="ALL">All Severities</option>
              <option value="Low">Low</option>
              <option value="Moderate">Moderate</option>
              <option value="Severe">Severe</option>
              <option value="Critical">Critical</option>
            </select>
          </div>
        </div>

        {/* Search Bar */}
        <div className="relative flex-1 sm:max-w-xs">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400 dark:text-ink-muted" />
          <input
            type="text"
            data-testid="search-scouting-input"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search location, pathogen, notes..."
            className="w-full rounded-lg border border-border-light bg-surface-light py-1.5 pl-8 pr-3 text-xs text-slate-900 placeholder:text-slate-400 focus:border-accent-500 focus:outline-none dark:border-border dark:bg-surface-dark dark:text-ink-primary"
          />
        </div>
      </div>

      {/* Field Disease Timeline & History List */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-ink-muted">
          <span>Showing {filteredLogs.length} of {logs.length} scouting records</span>
          {logs.length > 0 && (
            <button
              type="button"
              onClick={handleResetDefaults}
              className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-600 dark:hover:text-ink-primary"
              title="Reset records to default test dataset"
            >
              <RefreshCw size={11} />
              Reset Demo Log
            </button>
          )}
        </div>

        {filteredLogs.length === 0 ? (
          <div
            data-testid="scouting-empty-state"
            className="rounded-xl border border-dashed border-border-light p-8 text-center dark:border-border"
          >
            <Leaf size={28} className="mx-auto text-slate-400 dark:text-ink-muted opacity-50" />
            <h4 className="mt-2 text-sm font-semibold text-slate-800 dark:text-ink-primary">
              No matching scouting records
            </h4>
            <p className="mt-1 text-xs text-slate-500 dark:text-ink-muted">
              Adjust your filters or add a new scouting observation.
            </p>
          </div>
        ) : (
          <div className="space-y-3" data-testid="scouting-timeline-list">
            {filteredLogs.map((item) => {
              const isHealthy = (item.disease || '').toLowerCase().includes('healthy')
              const confPercent = Math.round((item.confidence ?? 0.9) * 100)

              return (
                <motion.div
                  key={item.id}
                  layout
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  data-testid={`scouting-entry-${item.id}`}
                  className="relative overflow-hidden rounded-xl border border-border-light bg-surface-light p-4 shadow-sm transition-all hover:border-accent-500/40 dark:border-border dark:bg-surface-dark"
                >
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    {/* Left: Crop, Pathogen, Severity, Location */}
                    <div className="space-y-1.5">
                      <div className="flex flex-wrap items-center gap-2">
                        {/* Crop Tag */}
                        <span className="inline-flex items-center gap-1 rounded-md bg-accent-500/10 px-2 py-0.5 text-xs font-semibold text-accent-700 dark:text-accent-300">
                          <Sprout size={12} />
                          {item.crop}
                        </span>

                        {/* Severity Badge */}
                        <span
                          className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-bold ${getSeverityBadgeStyle(
                            item.severity
                          )}`}
                        >
                          {isHealthy ? (
                            <ShieldCheck size={12} />
                          ) : (
                            <ShieldAlert size={12} />
                          )}
                          {item.severity} Severity
                        </span>

                        {/* Confidence */}
                        <span className="text-[11px] font-medium text-slate-400 dark:text-ink-muted">
                          {confPercent}% confidence
                        </span>
                      </div>

                      {/* Pathogen Title */}
                      <h3 className="font-display text-base font-bold text-slate-900 dark:text-ink-primary">
                        {item.disease}
                      </h3>

                      {/* Location & Time */}
                      <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-ink-muted">
                        <span className="inline-flex items-center gap-1 font-medium text-slate-700 dark:text-ink-secondary">
                          <MapPin size={13} className="text-accent-500" />
                          {item.location}
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <Clock size={13} />
                          {formatDate(item.timestamp)}
                        </span>
                      </div>

                      {/* Notes */}
                      {item.notes && (
                        <p className="pt-1 text-xs text-slate-600 dark:text-ink-secondary">
                          {item.notes}
                        </p>
                      )}

                      {/* Remedy Applied */}
                      {item.remedyApplied && (
                        <div className="pt-1 flex items-center gap-1.5 text-xs">
                          <span className="font-semibold text-slate-700 dark:text-ink-primary">Remedy Applied:</span>
                          <span className="rounded bg-slate-900/5 px-2 py-0.5 text-slate-800 dark:bg-white/5 dark:text-ink-secondary">
                            {item.remedyApplied}
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Right: Delete Action */}
                    <div className="flex items-center sm:self-start">
                      <button
                        type="button"
                        onClick={() => handleDelete(item.id)}
                        data-testid={`delete-scout-${item.id}`}
                        className="rounded-lg p-1.5 text-slate-400 transition hover:bg-rose-500/10 hover:text-rose-600 dark:hover:bg-rose-500/10 dark:hover:text-rose-400"
                        title="Delete scouting entry"
                        aria-label={`Delete scouting entry ${item.id}`}
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        )}
      </div>

      {/* Manual Add New Scout Entry Modal */}
      <AnimatePresence>
        {showAddModal && (
          <div
            data-testid="add-scout-modal-backdrop"
            className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-black/60 p-4 backdrop-blur-sm"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              className="relative w-full max-w-lg rounded-2xl border border-border-light bg-card-light p-6 shadow-2xl dark:border-border dark:bg-card"
            >
              <div className="flex items-center justify-between border-b border-border-light pb-3.5 dark:border-border">
                <div className="flex items-center gap-2">
                  <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-500/15 text-accent-600 dark:text-accent-400">
                    <Plus size={16} />
                  </span>
                  <h3 className="font-display text-base font-bold text-slate-900 dark:text-ink-primary">
                    Log Field Scouting Observation
                  </h3>
                </div>
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="rounded-lg p-1 text-slate-400 hover:text-slate-600 dark:hover:text-ink-primary"
                  aria-label="Close add scout modal"
                >
                  <X size={18} />
                </button>
              </div>

              <form onSubmit={handleCreateEntry} className="mt-4 space-y-4">
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {/* Crop */}
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 dark:text-ink-primary">
                      Crop Species *
                    </label>
                    <input
                      type="text"
                      required
                      data-testid="form-crop-input"
                      value={formCrop}
                      onChange={(e) => setFormCrop(e.target.value)}
                      placeholder="e.g. Tomato, Corn, Potato"
                      className="mt-1 w-full rounded-lg border border-border-light bg-surface-light px-3 py-1.5 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-border dark:bg-surface-dark dark:text-ink-primary"
                    />
                  </div>

                  {/* Diagnosed Disease / Condition */}
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 dark:text-ink-primary">
                      Diagnosed Condition *
                    </label>
                    <input
                      type="text"
                      required
                      data-testid="form-disease-input"
                      value={formDisease}
                      onChange={(e) => setFormDisease(e.target.value)}
                      placeholder="e.g. Early Blight, Powdery Mildew"
                      className="mt-1 w-full rounded-lg border border-border-light bg-surface-light px-3 py-1.5 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-border dark:bg-surface-dark dark:text-ink-primary"
                    />
                  </div>

                  {/* Severity */}
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 dark:text-ink-primary">
                      Severity Level
                    </label>
                    <select
                      data-testid="form-severity-select"
                      value={formSeverity}
                      onChange={(e) => setFormSeverity(e.target.value)}
                      className="mt-1 w-full rounded-lg border border-border-light bg-surface-light px-3 py-1.5 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-border dark:bg-surface-dark dark:text-ink-primary"
                    >
                      <option value="Low">Low</option>
                      <option value="Moderate">Moderate</option>
                      <option value="Severe">Severe</option>
                      <option value="Critical">Critical</option>
                    </select>
                  </div>

                  {/* Confidence */}
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 dark:text-ink-primary">
                      Confidence Score (%)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      data-testid="form-confidence-input"
                      value={formConfidence}
                      onChange={(e) => setFormConfidence(e.target.value)}
                      className="mt-1 w-full rounded-lg border border-border-light bg-surface-light px-3 py-1.5 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-border dark:bg-surface-dark dark:text-ink-primary"
                    />
                  </div>
                </div>

                {/* Location */}
                <div>
                  <label className="block text-xs font-semibold text-slate-700 dark:text-ink-primary">
                    Field Location / Zone / GPS *
                  </label>
                  <input
                    type="text"
                    required
                    data-testid="form-location-input"
                    value={formLocation}
                    onChange={(e) => setFormLocation(e.target.value)}
                    placeholder="e.g. North Orchard Block B, 37.7749° N, 122.4194° W"
                    className="mt-1 w-full rounded-lg border border-border-light bg-surface-light px-3 py-1.5 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-border dark:bg-surface-dark dark:text-ink-primary"
                  />
                </div>

                {/* Remedy Applied */}
                <div>
                  <label className="block text-xs font-semibold text-slate-700 dark:text-ink-primary">
                    Remedy / Treatment Applied
                  </label>
                  <input
                    type="text"
                    data-testid="form-remedy-input"
                    value={formRemedy}
                    onChange={(e) => setFormRemedy(e.target.value)}
                    placeholder="e.g. Copper Hydroxide 2.5g/L or Pending Assessment"
                    className="mt-1 w-full rounded-lg border border-border-light bg-surface-light px-3 py-1.5 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-border dark:bg-surface-dark dark:text-ink-primary"
                  />
                </div>

                {/* Notes */}
                <div>
                  <label className="block text-xs font-semibold text-slate-700 dark:text-ink-primary">
                    Agronomic Field Notes
                  </label>
                  <textarea
                    rows={2}
                    data-testid="form-notes-input"
                    value={formNotes}
                    onChange={(e) => setFormNotes(e.target.value)}
                    placeholder="Describe symptoms, infected row numbers, spread rate, soil moisture..."
                    className="mt-1 w-full rounded-lg border border-border-light bg-surface-light px-3 py-1.5 text-xs text-slate-900 focus:border-accent-500 focus:outline-none dark:border-border dark:bg-surface-dark dark:text-ink-primary"
                  />
                </div>

                {/* Modal Footer Buttons */}
                <div className="flex items-center justify-end gap-2 border-t border-border-light pt-3 dark:border-border">
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={() => setShowAddModal(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    variant="primary"
                    size="sm"
                    data-testid="submit-add-scout-button"
                  >
                    Save Scouting Entry
                  </Button>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  )
}
