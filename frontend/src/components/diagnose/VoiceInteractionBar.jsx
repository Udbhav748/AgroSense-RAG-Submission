import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Play,
  Pause,
  Square,
  Sparkles,
  AlertCircle,
  Radio,
  Activity,
  CheckCircle2,
  Globe,
  Languages,
} from 'lucide-react'
import { getAgronomicGuide, getSeverityInfo } from '../../utils/agronomyData'
import {
  SUPPORTED_LANGUAGES,
  VOICE_LANGUAGE_MAP,
  getVoiceLanguage,
  getLanguageConfig,
  t,
} from '../../utils/i18n'

/**
 * Builds a clear, spoken 24-48h emergency field action script from diagnosis and agronomic guides.
 */
export function buildEmergencyFieldProtocolText(diagnosis, customAnswer, language = 'en') {
  if (!diagnosis) return t('protocol.noDiagnosis', language)

  const crop = diagnosis.crop || 'crop'
  const disease = diagnosis.disease || 'condition'
  const confidence = diagnosis.confidence ? Math.round(diagnosis.confidence * 100) : 95
  const severity = getSeverityInfo(disease, diagnosis.confidence || 1.0)
  const guide = getAgronomicGuide(disease, crop)

  const isHealthy = disease.toLowerCase().includes('healthy')

  if (isHealthy) {
    if (language === 'en') {
      return `Field Health Report: Your ${crop} foliage is diagnosed as healthy with ${confidence}% certainty. Severity is low. Continue routine weekly scouting and balanced organic nutrition. No chemical fungicide is required.`
    }
    return t('protocol.healthyReport', language, { crop, confidence })
  }

  const bioRemedy = guide.organicRemedies?.bioFungicides?.[0] || 'Apply preventative bio-fungicide.'
  const culturalStep = guide.organicRemedies?.culturalPractices?.[0] || 'Prune infected foliage and avoid overhead watering.'
  const chemStep = guide.chemicalControl?.activeIngredients?.[0]
    ? `${guide.chemicalControl.activeIngredients[0].name} at ${guide.chemicalControl.activeIngredients[0].rate}`
    : 'Apply registered protectant fungicide.'
  const sprayInterval = guide.chemicalControl?.sprayInterval || 'Apply every 7 to 10 days during humid weather.'

  if (language === 'en') {
    return (
      `Emergency 24 to 48-Hour Field Protocol for ${crop} affected by ${disease}. ` +
      `Severity level: ${severity.level} with ${confidence}% visual certainty. ` +
      `Pathogen: ${guide.pathogen}. ` +
      `Immediate Step 1: Cultural Sanitation. ${culturalStep} ` +
      `Immediate Step 2: Biological Control. ${bioRemedy} ` +
      `Immediate Step 3: Chemical Treatment if infection persists. Apply ${chemStep}. ` +
      `Spray interval: ${sprayInterval} ` +
      `Safety Advisory: Always wear personal protective equipment and observe pre-harvest interval regulations.`
    )
  }

  const intro = t('protocol.emergencyIntro', language, { crop, disease })
  const sev = t('protocol.severityLevel', language, { level: severity.level, confidence })
  const pathogen = t('protocol.pathogenLabel', language, { pathogen: guide.pathogen })
  const step1 = t('protocol.step1Sanitation', language, { step: culturalStep })
  const step2 = t('protocol.step2Biological', language, { step: bioRemedy })
  const step3 = t('protocol.step3Chemical', language, { step: chemStep })
  const interval = t('protocol.sprayIntervalLabel', language, { interval: sprayInterval })
  const safety = t('protocol.safetyAdvisory', language)

  return `${intro} ${sev} ${pathogen} ${step1} ${step2} ${step3} ${interval} ${safety}`
}

/**
 * Hands-Free Microphone Dictation Button (Speech-to-Text).
 * Uses browser Web Speech API (webkitSpeechRecognition / SpeechRecognition).
 */
export function SpeechToTextButton({
  onTranscript,
  disabled = false,
  language = 'en',
  className = '',
  size = 'md',
  placeholder = 'Speak to add context or prompt...',
}) {
  const [isListening, setIsListening] = useState(false)
  const [isSupported, setIsSupported] = useState(true)
  const [errorMessage, setErrorMessage] = useState(null)
  const recognitionRef = useRef(null)

  const voiceLangCode = getVoiceLanguage(language)

  useEffect(() => {
    const SpeechRecognition =
      typeof window !== 'undefined' &&
      (window.SpeechRecognition || window.webkitSpeechRecognition)

    if (!SpeechRecognition) {
      setIsSupported(false)
      return
    }

    try {
      const recognition = new SpeechRecognition()
      recognition.continuous = false
      recognition.interimResults = true
      recognition.lang = voiceLangCode

      recognition.onstart = () => {
        setIsListening(true)
        setErrorMessage(null)
      }

      recognition.onresult = (event) => {
        let interimTranscript = ''
        let finalTranscript = ''

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const transcriptChunk = event.results[i][0]?.transcript || ''
          if (event.results[i].isFinal) {
            finalTranscript += transcriptChunk
          } else {
            interimTranscript += transcriptChunk
          }
        }

        const currentText = finalTranscript || interimTranscript
        if (currentText && onTranscript) {
          onTranscript(currentText.trim())
        }
      }

      recognition.onerror = (event) => {
        // Ignore aborted or no-speech quiet events gracefully
        if (event.error !== 'no-speech' && event.error !== 'aborted') {
          setErrorMessage(`Mic error: ${event.error}`)
        }
        setIsListening(false)
      }

      recognition.onend = () => {
        setIsListening(false)
      }

      recognitionRef.current = recognition
    } catch (e) {
      setIsSupported(false)
    }

    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort()
        } catch {
          // ignore
        }
      }
    }
  }, [onTranscript, voiceLangCode])

  const toggleListening = () => {
    if (disabled || !isSupported) return

    if (isListening) {
      try {
        recognitionRef.current?.stop()
      } catch {
        // ignore
      }
      setIsListening(false)
    } else {
      setErrorMessage(null)
      if (recognitionRef.current) {
        recognitionRef.current.lang = voiceLangCode
      }
      try {
        recognitionRef.current?.start()
      } catch (err) {
        // In case recognition was already started or needs restart
        try {
          recognitionRef.current?.stop()
          setTimeout(() => {
            if (recognitionRef.current) {
              recognitionRef.current.lang = voiceLangCode
              recognitionRef.current.start()
            }
          }, 100)
        } catch {
          setErrorMessage('Could not initialize microphone.')
          setIsListening(false)
        }
      }
    }
  }

  const isSmall = size === 'sm'

  return (
    <div className="relative inline-flex items-center">
      <motion.button
        type="button"
        data-testid="speech-to-text-mic-button"
        onClick={toggleListening}
        disabled={disabled || !isSupported}
        whileHover={{ scale: disabled || !isSupported ? 1 : 1.05 }}
        whileTap={{ scale: disabled || !isSupported ? 1 : 0.95 }}
        aria-label={
          !isSupported
            ? 'Voice input not supported in this browser'
            : isListening
            ? 'Stop voice recording'
            : 'Start hands-free voice dictation'
        }
        title={
          !isSupported
            ? 'Web Speech API is not supported in this browser'
            : isListening
            ? 'Listening... Click to stop'
            : `Voice dictation (${getLanguageConfig(language).nativeLabel})`
        }
        className={`relative flex items-center justify-center rounded-xl font-medium transition-all ${
          isSmall ? 'h-8 w-8 p-1 text-xs' : 'h-10 w-10 text-sm'
        } ${
          isListening
            ? 'bg-rose-500 text-white shadow-md shadow-rose-500/30 ring-2 ring-rose-400 ring-offset-2 dark:ring-offset-slate-900'
            : isSupported
            ? 'border border-border-light bg-slate-900/[0.04] text-slate-700 hover:border-accent-500/50 hover:bg-accent-500/10 hover:text-accent-600 dark:border-border dark:bg-white/[0.04] dark:text-ink-secondary dark:hover:border-accent-500/50 dark:hover:bg-white/10 dark:hover:text-ink-primary'
            : 'cursor-not-allowed opacity-40 border border-dashed border-slate-300 text-slate-400 dark:border-slate-700 dark:text-slate-600'
        } ${className}`}
      >
        {isListening ? (
          <>
            <motion.span
              animate={{ scale: [1, 1.35, 1], opacity: [0.7, 0, 0.7] }}
              transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
              className="absolute inset-0 rounded-xl bg-rose-500"
            />
            <Mic size={isSmall ? 15 : 18} className="relative z-10 animate-pulse text-white" />
          </>
        ) : (
          <Mic size={isSmall ? 15 : 18} />
        )}
      </motion.button>

      {/* Floating active speech status pill */}
      <AnimatePresence>
        {isListening && (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.95 }}
            className="absolute bottom-full left-1/2 mb-2 -translate-x-1/2 whitespace-nowrap rounded-full bg-slate-950 px-3 py-1 text-[11px] font-semibold text-rose-300 shadow-xl ring-1 ring-rose-500/40 backdrop-blur-md dark:bg-slate-900"
            role="status"
            aria-live="polite"
          >
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 animate-ping rounded-full bg-rose-500" />
              <span>{t('buttons.listening', language)} ({voiceLangCode})</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/**
 * Animated Soundwave Bar Component
 */
function SoundwaveBar({ isPlaying, isPaused }) {
  const bars = [16, 28, 12, 32, 22, 14, 26, 18, 30, 20]

  return (
    <div
      data-testid="audio-soundwave-bars"
      aria-hidden="true"
      className="flex h-7 items-center gap-0.5 px-2"
    >
      {bars.map((height, idx) => (
        <motion.span
          key={idx}
          animate={
            isPlaying && !isPaused
              ? {
                  scaleY: [0.3, 1, 0.4, 0.9, 0.3],
                }
              : { scaleY: 0.3 }
          }
          transition={{
            duration: 0.8 + (idx % 3) * 0.2,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: (idx * 0.08) % 0.4,
          }}
          style={{ height: `${height}px`, transformOrigin: 'bottom' }}
          className={`w-1 rounded-full transition-colors ${
            isPlaying && !isPaused
              ? 'bg-accent-500 dark:bg-accent-400'
              : 'bg-slate-300 dark:bg-white/20'
          }`}
        />
      ))}
    </div>
  )
}

/**
 * Field Protocol Text-to-Speech Audio Player (TTS).
 * Provides "[ Listen to 24h Field Protocol ]" with Play, Pause, Stop, and Soundwave visualization.
 */
export function FieldProtocolAudioPlayer({
  diagnosis,
  customText,
  language = 'en',
  title = '24h Field Treatment Protocol',
  className = '',
}) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [isSupported, setIsSupported] = useState(true)
  const [speechRate, setSpeechRate] = useState(0.95) // Clear pace for noisy fields
  const utteranceRef = useRef(null)

  const voiceLangCode = getVoiceLanguage(language)

  useEffect(() => {
    if (typeof window === 'undefined' || !window.speechSynthesis) {
      setIsSupported(false)
    }

    return () => {
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        window.speechSynthesis.cancel()
      }
    }
  }, [])

  const fullTextToRead =
    customText || buildEmergencyFieldProtocolText(diagnosis, null, language)

  const handlePlay = () => {
    if (!isSupported || typeof window === 'undefined' || !window.speechSynthesis) return

    if (isPaused) {
      window.speechSynthesis.resume()
      setIsPaused(false)
      setIsPlaying(true)
      return
    }

    window.speechSynthesis.cancel()

    const utterance = new SpeechSynthesisUtterance(fullTextToRead)
    utterance.rate = speechRate
    utterance.pitch = 1.0
    utterance.lang = voiceLangCode

    utterance.onstart = () => {
      setIsPlaying(true)
      setIsPaused(false)
    }

    utterance.onpause = () => {
      setIsPaused(true)
    }

    utterance.onresume = () => {
      setIsPaused(false)
    }

    utterance.onend = () => {
      setIsPlaying(false)
      setIsPaused(false)
    }

    utterance.onerror = () => {
      setIsPlaying(false)
      setIsPaused(false)
    }

    utteranceRef.current = utterance
    window.speechSynthesis.speak(utterance)
  }

  const handlePause = () => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return
    window.speechSynthesis.pause()
    setIsPaused(true)
  }

  const handleStop = () => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return
    window.speechSynthesis.cancel()
    setIsPlaying(false)
    setIsPaused(false)
  }

  if (!isSupported) {
    return (
      <div className="inline-flex items-center gap-2 rounded-xl border border-border-light bg-slate-900/[0.02] px-3 py-1.5 text-xs text-slate-400 dark:border-border dark:bg-white/[0.02] dark:text-ink-muted">
        <VolumeX size={15} />
        <span>Audio narration unsupported in this browser</span>
      </div>
    )
  }

  return (
    <div
      data-testid="field-protocol-audio-player"
      className={`flex flex-wrap items-center justify-between gap-3 rounded-xl border border-accent-500/20 bg-accent-500/5 p-3 text-xs dark:border-accent-500/25 dark:bg-accent-500/10 ${className}`}
    >
      {/* Label and Audio info */}
      <div className="flex items-center gap-2.5">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-500/20 text-accent-600 dark:text-accent-400">
          <Volume2 size={17} />
        </span>
        <div>
          <div className="flex items-center gap-2 font-semibold text-slate-900 dark:text-ink-primary">
            <span>{title}</span>
            {isPlaying && !isPaused && (
              <span className="inline-flex items-center gap-1 rounded-full bg-accent-500/20 px-2 py-0.2 text-[10px] font-bold text-accent-700 dark:text-accent-300">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-500" />
                Reading Aloud ({voiceLangCode})
              </span>
            )}
          </div>
          <p className="text-[11px] text-slate-500 dark:text-ink-muted">
            {t('protocol.subtitle', language)}
          </p>
        </div>
      </div>

      {/* Controls & Soundwave Visualization */}
      <div className="flex items-center gap-2">
        {/* Soundwave */}
        <SoundwaveBar isPlaying={isPlaying} isPaused={isPaused} />

        {/* Play / Resume Button */}
        {!isPlaying || isPaused ? (
          <button
            type="button"
            data-testid="voice-play-protocol-button"
            onClick={handlePlay}
            aria-label="Listen to 24h Field Protocol"
            className="flex items-center gap-1.5 rounded-lg bg-accent-600 px-3 py-1.5 font-semibold text-white shadow-sm transition-all hover:bg-accent-700 active:scale-95 dark:bg-accent-500 dark:hover:bg-accent-600"
          >
            <Play size={14} className="fill-current" />
            <span>{isPaused ? t('buttons.resume', language) : t('buttons.listenProtocol', language)}</span>
          </button>
        ) : (
          <button
            type="button"
            data-testid="voice-pause-protocol-button"
            onClick={handlePause}
            aria-label="Pause Field Protocol Audio"
            className="flex items-center gap-1.5 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-1.5 font-semibold text-amber-700 transition-all hover:bg-amber-500/20 active:scale-95 dark:text-amber-300"
          >
            <Pause size={14} className="fill-current" />
            <span>{t('buttons.pause', language)}</span>
          </button>
        )}

        {/* Stop Button */}
        {isPlaying && (
          <button
            type="button"
            data-testid="voice-stop-protocol-button"
            onClick={handleStop}
            aria-label="Stop Field Protocol Audio"
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-rose-500/30 bg-rose-500/10 text-rose-600 transition-all hover:bg-rose-500/20 active:scale-95 dark:text-rose-400"
            title="Stop Audio"
          >
            <Square size={13} className="fill-current" />
          </button>
        )}
      </div>
    </div>
  )
}

/**
 * Language Selector Dropdown Component:
 * [ English | Español | हिन्दी | Português | Français | Kiswahili ]
 */
export function LanguageSelectorDropdown({
  value,
  language,
  onChange,
  className = '',
  size = 'md',
  disabled = false,
}) {
  const activeValue = value || language || 'en'

  return (
    <div className={`inline-flex items-center gap-1.5 ${className}`}>
      <Languages size={size === 'sm' ? 14 : 16} className="text-accent-600 dark:text-accent-400 shrink-0" />
      <select
        data-testid="voice-language-selector"
        aria-label="Select voice and text language"
        value={activeValue}
        disabled={disabled}
        onChange={(e) => onChange?.(e.target.value)}
        className="rounded-lg border border-border-light bg-white/80 px-2.5 py-1 text-xs font-medium text-slate-800 outline-none transition-colors focus:border-accent-500 focus:ring-2 focus:ring-accent-500/30 dark:border-border dark:bg-card dark:text-ink-primary disabled:opacity-50"
      >
        {SUPPORTED_LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.nativeLabel} ({lang.label})
          </option>
        ))}
      </select>
    </div>
  )
}

/**
 * Composite Voice Interaction Bar offering hands-free STT dictation,
 * TTS field audio, and multilingual language selection.
 */
export default function VoiceInteractionBar({
  diagnosis,
  customText,
  language = 'en',
  onLanguageChange,
  onTranscript,
  showDictation = true,
  showPlayer = true,
  showLanguageSelector = true,
  className = '',
}) {
  const [selectedLanguage, setSelectedLanguage] = useState(language)

  useEffect(() => {
    setSelectedLanguage(language)
  }, [language])

  const handleLangChange = (newLang) => {
    setSelectedLanguage(newLang)
    onLanguageChange?.(newLang)
  }

  return (
    <div
      data-testid="voice-interaction-bar"
      className={`space-y-3 ${className}`}
    >
      {/* Top Bar with Language Selector */}
      {showLanguageSelector && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-border-light bg-slate-900/[0.02] p-2.5 text-xs dark:border-border dark:bg-white/[0.02]">
          <div className="flex items-center gap-2 text-slate-600 dark:text-ink-secondary">
            <Globe size={15} className="text-accent-500" />
            <span className="font-semibold">{t('buttons.selectLanguage', selectedLanguage)}:</span>
          </div>
          <LanguageSelectorDropdown
            value={selectedLanguage}
            onChange={handleLangChange}
            size="sm"
          />
        </div>
      )}

      {showPlayer && diagnosis && (
        <FieldProtocolAudioPlayer
          diagnosis={diagnosis}
          customText={customText}
          language={selectedLanguage}
        />
      )}

      {showDictation && onTranscript && (
        <div className="flex items-center justify-between rounded-xl border border-border-light bg-slate-900/[0.02] p-3 text-xs dark:border-border dark:bg-white/[0.02]">
          <div className="flex items-center gap-2 text-slate-600 dark:text-ink-secondary">
            <Radio size={15} className="text-accent-500" />
            <span>{t('buttons.voiceDictation', selectedLanguage)}</span>
          </div>
          <SpeechToTextButton
            onTranscript={onTranscript}
            language={selectedLanguage}
          />
        </div>
      )}
    </div>
  )
}
