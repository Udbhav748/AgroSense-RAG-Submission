import { motion } from 'framer-motion'
import PredictionHeroCard from './PredictionHeroCard'
import TreatmentPlanTabs from './TreatmentPlanTabs'
import SprayDosageCalculator from './SprayDosageCalculator'

/**
 * Composite Diagnosis Result Hub containing:
 * - Prediction Hero Card
 * - 5-Tab Treatment & Agronomic Plan
 * - Interactive Spray Dosage & Mix Volume Calculator Widget
 */
export default function DiagnosisResult({
  result,
  previewUrl,
  onReset,
  query,
  isStreaming = false,
  language = 'en',
  onLanguageChange,
}) {
  const { diagnosis, answer, sources, processing_time, session_id } = result

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className="space-y-6"
    >
      {/* 1. Prediction Hero Card */}
      <PredictionHeroCard
        diagnosis={diagnosis}
        previewUrl={previewUrl}
        sessionId={session_id}
        processingTime={processing_time}
        onReset={onReset}
        language={language}
        onLanguageChange={onLanguageChange}
      />

      {/* 2. Tabbed Treatment Plan */}
      <TreatmentPlanTabs
        diagnosis={diagnosis}
        answer={answer}
        sources={sources}
        query={query}
        isStreaming={isStreaming}
      />

      {/* 3. Spray Dosage Calculator Widget */}
      <SprayDosageCalculator
        defaultDisease={diagnosis?.disease}
        defaultCrop={diagnosis?.crop}
      />

      <p className="text-center text-[11px] text-slate-400 dark:text-ink-muted">
        {processing_time != null ? `Processed in ${typeof processing_time === 'number' ? processing_time.toFixed(2) : processing_time}s · ` : ''}
        Agronomic advice is generated from LeafSense hybrid vision models and retrieved extension manuals. Always verify localized regulations and follow pesticide label instructions.
      </p>
    </motion.div>
  )
}
