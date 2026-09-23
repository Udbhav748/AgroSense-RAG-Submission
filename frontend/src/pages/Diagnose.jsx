import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  Calculator,
  Camera,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileText,
  FlaskConical,
  Info,
  Leaf,
  MapPin,
  RefreshCw,
  ScanLine,
  ShieldAlert,
  Sparkles,
  Sprout,
  UploadCloud,
} from 'lucide-react'
import Button from '../components/ui/Button'
import ServiceHealthBanner from '../components/diagnose/ServiceHealthBanner'
import UploadCameraArea from '../components/diagnose/UploadCameraArea'
import DiagnosisResult from '../components/diagnose/DiagnosisResult'
import SprayDosageCalculator from '../components/diagnose/SprayDosageCalculator'
import FieldScoutingLog from '../components/diagnose/FieldScoutingLog'
import useDiagnose from '../hooks/useDiagnose'

export default function Diagnose() {
  const {
    image,
    previewUrl,
    query,
    engine,
    setEngine,
    language,
    setLanguage,
    status,
    isStreaming,
    result,
    errorMessage,
    isLeafSenseOnline,
    isCheckingHealth,
    checkHealth,
    selectImage,
    clearImage,
    setQueryText,
    analyze,
    reset,
  } = useDiagnose()

  const [showStandaloneCalculator, setShowStandaloneCalculator] = useState(false)
  const [showScoutingHistory, setShowScoutingHistory] = useState(false)

  const isLowConfidenceOrOOD =
    (result?.diagnosis?.confidence != null && result?.diagnosis?.confidence < 0.45) ||
    Boolean(result?.diagnosis?.low_confidence) ||
    Boolean(result?.diagnosis?.is_non_leaf) ||
    Boolean(result?.diagnosis?.is_uncertain) ||
    (result?.diagnosis?.disease && /uncertain|non-leaf|non_leaf|unknown|invalid/i.test(result.diagnosis.disease))

  return (
    <div className="mx-auto max-w-4xl space-y-6 py-4">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-2 rounded-full border border-accent-500/20 bg-accent-500/10 px-3 py-1 text-xs font-semibold text-accent-700 dark:text-accent-300">
          <Sprout size={14} className="text-accent-600 dark:text-accent-400" />
          <span>Agricultural Pathology & RAG Treatment Hub</span>
        </div>
        <h1 className="font-display text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl dark:text-ink-primary">
          Plant Leaf Disease Diagnostic & Treatment Hub
        </h1>
        <p className="mx-auto max-w-2xl text-xs sm:text-sm text-slate-600 dark:text-ink-secondary">
          Upload or capture a leaf photo. Switch between our custom LeafSense deep vision network (port 8001), multimodal zero-shot vision, or the cascaded hybrid arbiter.
        </p>
      </div>

      {/* 1. Service Health Warning Indicator (Port 8001) */}
      <ServiceHealthBanner
        isOnline={isLeafSenseOnline}
        isChecking={isCheckingHealth}
        onRecheck={checkHealth}
      />

      {/* Main Workflow Area */}
      <AnimatePresence mode="wait">
        {/* State: Idle / Preview / Analyzing (before diagnosis arrives) */}
        {(status === 'idle' || status === 'preview' || (status === 'analyzing' && !result?.diagnosis)) && (
          <motion.div
            key="upload-section"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.25 }}
            className="space-y-6"
          >
            <UploadCameraArea
              image={image}
              previewUrl={previewUrl}
              query={query}
              engine={engine}
              language={language}
              status={status}
              onSelectImage={selectImage}
              onClearImage={clearImage}
              onQueryChange={setQueryText}
              onEngineChange={setEngine}
              onLanguageChange={setLanguage}
              onAnalyze={analyze}
            />

            {/* Standalone Spray Calculator Toggle when not diagnosing */}
            {status === 'idle' && (
              <div className="pt-2">
                <div className="flex items-center justify-between border-t border-border-light pt-4 dark:border-border">
                  <div className="flex items-center gap-2 text-xs text-slate-600 dark:text-ink-secondary">
                    <Calculator size={15} className="text-accent-500" />
                    <span>Need to calculate chemical tank mix dosages without a photo?</span>
                  </div>
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={() => setShowStandaloneCalculator((prev) => !prev)}
                  >
                    {showStandaloneCalculator ? 'Hide Calculator' : 'Open Spray Calculator'}
                  </Button>
                </div>

                {showStandaloneCalculator && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-4"
                  >
                    <SprayDosageCalculator />
                  </motion.div>
                )}
              </div>
            )}
          </motion.div>
        )}

        {/* State: Error */}
        {status === 'error' && (
          <motion.div
            key="error-section"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="space-y-4"
          >
            <div className="panel rounded-panel border border-rose-500/30 bg-rose-500/5 p-6 text-center shadow-soft dark:border-rose-500/30 dark:bg-rose-500/10">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-rose-500/20 text-rose-600 dark:text-rose-400">
                <AlertCircle size={24} />
              </div>
              <h3 className="mt-3 font-display text-base font-bold text-slate-900 dark:text-ink-primary">
                Diagnosis Request Could Not Be Completed
              </h3>
              <p className="mx-auto mt-1.5 max-w-md text-xs sm:text-sm text-slate-600 dark:text-ink-secondary">
                {errorMessage}
              </p>

              <div className="mt-5 flex flex-wrap items-center justify-center gap-3">
                <Button variant="primary" icon={RefreshCw} onClick={analyze}>
                  Try Again
                </Button>
                <Button variant="ghost" onClick={reset}>
                  Select Another Leaf Photo
                </Button>
              </div>
            </div>
          </motion.div>
        )}

        {/* State: Streaming / Success -> Hero Card, 5-Tab Treatment Plan, and Calculator */}
        {(status === 'success' || status === 'streaming') && result?.diagnosis && (
          <motion.div
            key="result-section"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="space-y-6"
          >
            {/* OOD Non-Leaf Gatekeeper Warning Banner */}
            {isLowConfidenceOrOOD && (
              <div
                data-testid="ood-nonleaf-banner"
                className="flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-950 dark:border-rose-500/30 dark:bg-rose-500/15 dark:text-rose-200"
              >
                <AlertTriangle size={18} className="mt-0.5 shrink-0 text-rose-600 dark:text-rose-400" />
                <div className="space-y-1">
                  <p className="font-bold text-rose-900 dark:text-rose-100">
                    Low Visual Confidence / Non-Plant Detected. Please ensure the photo is well-lit and clearly shows an infected crop leaf.
                  </p>
                  <p className="text-[11px] text-rose-800/80 dark:text-rose-200/80">
                    The leaf diagnosis confidence is below 45% or the image was flagged as non-plant. For critical field decisions, re-take the photo under natural diffuse lighting.
                  </p>
                </div>
              </div>
            )}

            <DiagnosisResult
              result={result}
              previewUrl={previewUrl}
              onReset={reset}
              query={query}
              isStreaming={isStreaming || status === 'streaming'}
              language={language}
              onLanguageChange={setLanguage}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Field Scouting History Expandable Panel */}
      <div className="border-t border-border-light pt-4 dark:border-border" data-testid="field-scouting-expandable-panel">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-slate-600 dark:text-ink-secondary">
            <MapPin size={15} className="text-accent-500" />
            <span className="font-medium">Field Scouting History & Outbreak Tracking</span>
          </div>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            icon={showScoutingHistory ? ChevronUp : ChevronDown}
            onClick={() => setShowScoutingHistory((prev) => !prev)}
            data-testid="toggle-scouting-history-panel"
          >
            {showScoutingHistory ? 'Hide Scouting History' : 'Field Scouting History'}
          </Button>
        </div>

        {showScoutingHistory && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4"
          >
            <FieldScoutingLog currentDiagnosis={result?.diagnosis} />
          </motion.div>
        )}
      </div>
    </div>
  )
}
