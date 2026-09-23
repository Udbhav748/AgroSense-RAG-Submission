import { useState } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Eye,
  FileDown,
  FileText,
  Flame,
  Layers,
  Leaf,
  MapPin,
  MessageCircle,
  Printer,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
} from 'lucide-react'
import Button from '../ui/Button'
import { getSeverityInfo } from '../../utils/agronomyData'
import { addScoutingEntry } from '../../utils/scoutingStorage'
import PrescriptionWorkOrderModal from './PrescriptionWorkOrderModal'
import { FieldProtocolAudioPlayer } from './VoiceInteractionBar'

function confidenceColor(confidence) {
  if (confidence >= 0.85) return 'text-emerald-600 dark:text-emerald-400'
  if (confidence >= 0.65) return 'text-amber-600 dark:text-amber-400'
  return 'text-rose-600 dark:text-rose-400'
}

function confidenceBarColor(confidence) {
  if (confidence >= 0.85) return 'bg-emerald-500'
  if (confidence >= 0.65) return 'bg-amber-500'
  return 'bg-rose-500'
}

/**
 * Prediction Hero Card displaying detected crop, disease diagnosis,
 * severity level badge, animated confidence score bar, OOD warnings,
 * explainability heatmap overlay with opacity slider, visual metric badges,
 * memory bridge to chat, and prescription work order generator.
 */
export default function PredictionHeroCard({
  diagnosis,
  previewUrl,
  sessionId,
  processingTime,
  onReset,
  onSaveToLog,
  language = 'en',
  onLanguageChange,
}) {
  const navigate = useNavigate()
  const [showPrescriptionModal, setShowPrescriptionModal] = useState(false)
  const [isSavedToLog, setIsSavedToLog] = useState(false)
  const [viewMode, setViewMode] = useState('heatmap') // 'original' | 'heatmap'
  const [heatmapOpacity, setHeatmapOpacity] = useState(80) // 0 to 100

  const confidence = diagnosis?.confidence ?? 0
  const confidencePercent = Math.round(confidence * 100)
  const isHealthy = (diagnosis?.disease || '').toLowerCase().includes('healthy')
  const severity = getSeverityInfo(diagnosis?.disease, confidence)

  const hasHeatmap = Boolean(diagnosis?.heatmap_base64)
  const heatmapSrc = hasHeatmap
    ? diagnosis.heatmap_base64.startsWith('data:')
      ? diagnosis.heatmap_base64
      : `data:image/png;base64,${diagnosis.heatmap_base64}`
    : null

  const displayImgUrl = previewUrl || heatmapSrc

  const isLowConfidenceOrOOD =
    confidence < 0.45 ||
    Boolean(diagnosis?.low_confidence) ||
    Boolean(diagnosis?.is_non_leaf) ||
    Boolean(diagnosis?.is_uncertain) ||
    (diagnosis?.disease && /uncertain|non-leaf|non_leaf|unknown|invalid/i.test(diagnosis.disease))

  const handleContinueChat = () => {
    const crop = diagnosis?.crop || 'crop'
    const disease = diagnosis?.disease || 'condition'
    const sev = severity?.level || 'Moderate'

    const searchParams = new URLSearchParams()
    if (diagnosis?.crop) searchParams.set('crop', diagnosis.crop)
    if (diagnosis?.disease) searchParams.set('disease', diagnosis.disease)
    if (severity?.level) searchParams.set('severity', severity.level)
    if (language) searchParams.set('lang', language)

    const queryString = searchParams.toString() ? `?${searchParams.toString()}` : ''

    navigate(`/chat${queryString}`, {
      state: {
        sessionId,
        crop,
        disease,
        severity: sev,
        language,
      },
    })
  }

  const handleSaveToFieldLog = () => {
    if (!diagnosis) return
    const entry = addScoutingEntry({
      crop: diagnosis.crop || 'Crop',
      disease: diagnosis.disease || 'Unknown Condition',
      severity: severity?.level || 'Moderate',
      confidence: confidence,
      location: 'North Orchard Block B (Scouted)',
      notes: `Automated diagnosis saved from LeafSense hybrid vision (${diagnosis.raw_class || 'vision'})`,
      remedyApplied: 'Standard Treatment Recommended',
    })
    setIsSavedToLog(true)
    onSaveToLog?.(entry)
    setTimeout(() => {
      setIsSavedToLog(false)
    }, 2500)
  }

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        className="panel relative overflow-hidden rounded-panel border border-border-light p-6 shadow-soft dark:border-border"
      >
        {/* Background ambient gradient glow */}
        <div
          className={`pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full blur-3xl opacity-20 ${
            isHealthy ? 'bg-emerald-500' : severity.level === 'Severe' ? 'bg-rose-500' : 'bg-amber-500'
          }`}
        />

        <div className="relative space-y-5">
          {/* Top Tag & Severity / Metric Badges */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-500/15 text-accent-600 dark:text-accent-400">
                <Leaf size={15} strokeWidth={2} />
              </span>
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-ink-muted">
                {diagnosis?.engine === 'gemini_vision'
                  ? 'Multimodal Vision (Gemini 1.5 Flash)'
                  : diagnosis?.engine === 'hybrid_consensus'
                  ? 'Hybrid Consensus Arbiter (Verified)'
                  : 'LeafSense Hybrid CNN+ViT (Port 8001)'}
              </span>
            </div>

            {/* Badges Cluster */}
            <div className="flex flex-wrap items-center gap-2">
              {/* Severity Level Badge */}
              <div
                data-testid="severity-badge"
                className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${severity.badge}`}
              >
                {isHealthy ? (
                  <ShieldCheck size={14} className="shrink-0" />
                ) : severity.level === 'Severe' ? (
                  <ShieldAlert size={14} className="shrink-0" />
                ) : (
                  <AlertTriangle size={14} className="shrink-0" />
                )}
                <span>Severity: {severity.level}</span>
              </div>

              {/* Infected Leaf Area Badge */}
              {diagnosis?.infected_area_percentage != null && (
                <div
                  data-testid="infected-area-badge"
                  className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-xs font-semibold text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/15 dark:text-amber-300"
                >
                  <Activity size={13} className="shrink-0" />
                  <span>Infected Leaf Area: {diagnosis.infected_area_percentage}%</span>
                </div>
              )}

              {/* Estimated Lesions Count Badge */}
              {diagnosis?.lesion_count != null && (
                <div
                  data-testid="lesion-count-badge"
                  className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/30 bg-rose-500/10 px-3 py-1 text-xs font-semibold text-rose-700 dark:border-rose-500/30 dark:bg-rose-500/15 dark:text-rose-300"
                >
                  <Layers size={13} className="shrink-0" />
                  <span>Estimated Lesions: {diagnosis.lesion_count} spots</span>
                </div>
              )}
            </div>
          </div>

          {/* OOD Non-Leaf Gatekeeper Warning Banner */}
          {isLowConfidenceOrOOD && (
            <div
              data-testid="ood-gatekeeper-banner"
              className="flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-950 dark:border-rose-500/30 dark:bg-rose-500/15 dark:text-rose-200"
            >
              <AlertCircle size={18} className="mt-0.5 shrink-0 text-rose-600 dark:text-rose-400" />
              <div className="space-y-1">
                <p className="font-bold text-rose-900 dark:text-rose-100">
                  Low Visual Confidence / Non-Plant Detected. Please ensure the photo is well-lit and clearly shows an infected crop leaf.
                </p>
                <p className="text-[11px] text-rose-800/80 dark:text-rose-200/80">
                  The model classification confidence is low ({confidencePercent}%) or the uploaded texture did not correlate with standardized foliar pathology patterns. For accurate agronomic prescriptions, please upload a close-up, sharp photo of the infected leaf.
                </p>
              </div>
            </div>
          )}

          {/* Visual Heatmap / Photo Saliency Explorer */}
          {(displayImgUrl || heatmapSrc) && (
            <div className="overflow-hidden rounded-xl border border-border-light bg-slate-900/[0.02] p-3 sm:p-4 dark:border-border dark:bg-white/[0.02]">
              {/* Saliency Controls Bar */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border-light dark:border-border">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-slate-800 dark:text-ink-primary">
                    Visual Saliency & Pathology Heatmap:
                  </span>
                </div>

                {/* View Mode Toggle: [ Original Photo | Heatmap Lesion Overlay ] */}
                <div className="flex items-center gap-1.5 rounded-lg border border-border-light bg-slate-900/5 p-1 dark:border-border dark:bg-white/5">
                  <button
                    type="button"
                    data-testid="toggle-original-photo"
                    onClick={() => setViewMode('original')}
                    className={`rounded-md px-2.5 py-1 text-xs font-medium transition-all ${
                      viewMode === 'original'
                        ? 'bg-white text-slate-900 shadow-sm dark:bg-accent-600 dark:text-white'
                        : 'text-slate-600 hover:text-slate-900 dark:text-ink-muted dark:hover:text-white'
                    }`}
                  >
                    Original Photo
                  </button>
                  <button
                    type="button"
                    data-testid="toggle-heatmap-overlay"
                    onClick={() => setViewMode('heatmap')}
                    className={`rounded-md px-2.5 py-1 text-xs font-medium transition-all ${
                      viewMode === 'heatmap'
                        ? 'bg-white text-slate-900 shadow-sm dark:bg-accent-600 dark:text-white'
                        : 'text-slate-600 hover:text-slate-900 dark:text-ink-muted dark:hover:text-white'
                    }`}
                  >
                    Heatmap Lesion Overlay
                  </button>
                </div>
              </div>

              {/* Heatmap Opacity Slider & Color Legend */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-3 pb-2 text-xs">
                {/* Opacity Slider */}
                <div className="flex items-center gap-2.5">
                  <Sliders size={14} className="text-accent-500 shrink-0" />
                  <label htmlFor="heatmap-opacity-slider" className="font-medium text-slate-600 dark:text-ink-secondary">
                    Overlay Opacity:
                  </label>
                  <input
                    id="heatmap-opacity-slider"
                    data-testid="heatmap-opacity-slider"
                    type="range"
                    min="0"
                    max="100"
                    value={heatmapOpacity}
                    onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                    disabled={viewMode === 'original'}
                    className="h-1.5 w-24 sm:w-32 cursor-pointer appearance-none rounded-lg bg-slate-200 accent-accent-500 disabled:opacity-40 dark:bg-slate-700"
                  />
                  <span className="w-8 font-mono text-[11px] text-slate-500 dark:text-ink-muted">
                    {viewMode === 'original' ? '0%' : `${heatmapOpacity}%`}
                  </span>
                </div>

                {/* Heatmap Legend */}
                <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-600 dark:text-ink-secondary">
                  <span className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 ring-1 ring-emerald-500/40" />
                    <span>Healthy Tissue</span>
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-amber-400 ring-1 ring-amber-400/40" />
                    <span>Chlorotic Margins</span>
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-rose-500 ring-1 ring-rose-500/40" />
                    <span>Active Lesion Centers</span>
                  </span>
                </div>
              </div>

              {/* Image Viewport with Blended Layering */}
              <div className="relative mt-2 flex max-h-80 w-full items-center justify-center overflow-hidden rounded-lg bg-slate-950/40 p-2">
                {/* Base Original Photo */}
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Original leaf"
                    className="max-h-72 w-full rounded-md object-contain"
                  />
                ) : heatmapSrc ? (
                  <img
                    src={heatmapSrc}
                    alt="Leaf pathology"
                    className="max-h-72 w-full rounded-md object-contain"
                  />
                ) : null}

                {/* Heatmap Overlay Layer with Dynamic Alpha */}
                {heatmapSrc && previewUrl && (
                  <img
                    src={heatmapSrc}
                    alt="Heatmap lesion overlay"
                    style={{
                      opacity: viewMode === 'original' ? 0 : heatmapOpacity / 100,
                    }}
                    className="pointer-events-none absolute inset-0 m-auto max-h-72 w-full rounded-md object-contain transition-opacity duration-200"
                  />
                )}
              </div>
            </div>
          )}

          {/* Hero Plant & Disease Titles */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-12 sm:items-end">
            <div className="space-y-1 sm:col-span-8">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-ink-muted">
                Crop Species: <span className="font-semibold capitalize text-slate-800 dark:text-ink-primary">{diagnosis?.crop || 'Plant'}</span>
              </p>
              <h2 className="font-display text-2xl font-bold capitalize tracking-tight text-slate-900 sm:text-3xl dark:text-ink-primary">
                {diagnosis?.disease || 'Condition Undetermined'}
              </h2>
              <p className="text-xs text-slate-600 dark:text-ink-secondary">
                {severity.description}
              </p>
            </div>

            {/* Confidence Score Display */}
            <div className="rounded-xl border border-border-light bg-slate-900/[0.03] p-3 text-right sm:col-span-4 dark:border-border dark:bg-white/[0.02]">
              <span className="text-[11px] font-medium uppercase tracking-wider text-slate-500 dark:text-ink-muted">
                Classification Confidence
              </span>
              <div data-testid="confidence-score-value" className={`font-display text-3xl font-extrabold ${confidenceColor(confidence)}`}>
                {confidencePercent}%
              </div>
              <span className="text-[10px] text-slate-400 dark:text-ink-muted">
                Deep CNN & ViT Ensemble
              </span>
            </div>
          </div>

          {/* Animated Confidence Bar */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs text-slate-500 dark:text-ink-muted">
              <span>Model certainty score</span>
              <span className="font-medium">{confidencePercent}%</span>
            </div>
            <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-900/5 dark:bg-white/[0.06]">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${confidencePercent}%` }}
                transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1], delay: 0.1 }}
                className={`h-full rounded-full ${confidenceBarColor(confidence)}`}
              />
            </div>
          </div>

          {/* Raw model class & metadata info */}
          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border-light pt-3 text-[11px] text-slate-500 dark:border-border dark:text-ink-muted">
            <div>
              Raw Model Class: <code className="font-mono font-semibold text-slate-700 dark:text-ink-secondary">{diagnosis?.raw_class || 'N/A'}</code>
            </div>
            {processingTime != null && (
              <div>
                Processed in {typeof processingTime === 'number' ? processingTime.toFixed(2) : processingTime}s
              </div>
            )}
          </div>

          {/* Emergency 24-48h Field Protocol Voice Narration Player */}
          <FieldProtocolAudioPlayer
            diagnosis={diagnosis}
            title="24h Field Emergency Protocol Audio"
            language={language}
            onLanguageChange={onLanguageChange}
          />

          {/* Fast Action Buttons */}
          <div className="flex flex-wrap items-center gap-3 pt-1">
            {/* Primary Action Button: Continue in Agronomic Chat */}
            <Button
              type="button"
              variant="primary"
              icon={MessageCircle}
              onClick={handleContinueChat}
              disabled={!sessionId}
              data-testid="continue-in-chat-button"
            >
              Continue in Agronomic Chat
            </Button>

            {/* Download Spray Prescription Work Order */}
            <Button
              type="button"
              variant="secondary"
              icon={FileText}
              onClick={() => setShowPrescriptionModal(true)}
              data-testid="download-prescription-button"
            >
              Download Spray Prescription Work Order
            </Button>

            {/* Quick Action: Save to Field Log */}
            <Button
              type="button"
              variant="secondary"
              icon={isSavedToLog ? CheckCircle2 : MapPin}
              onClick={handleSaveToFieldLog}
              data-testid="save-to-field-log-button"
              className={isSavedToLog ? 'border-emerald-500/40 text-emerald-600 dark:text-emerald-400' : ''}
            >
              {isSavedToLog ? 'Saved to Field Log ✓' : 'Save to Field Log'}
            </Button>

            {/* Reset / Diagnose Another Leaf */}
            <Button
              type="button"
              variant="ghost"
              icon={RefreshCw}
              onClick={onReset}
            >
              Diagnose Another Leaf
            </Button>
          </div>
        </div>
      </motion.div>

      {/* Prescription Work Order Modal */}
      <PrescriptionWorkOrderModal
        isOpen={showPrescriptionModal}
        onClose={() => setShowPrescriptionModal(false)}
        diagnosis={diagnosis}
        severity={severity}
      />
    </>
  )
}
