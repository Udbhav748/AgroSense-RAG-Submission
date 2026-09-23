import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Camera, Cpu, ImagePlus, Leaf, RefreshCw, ScanLine, Sliders, UploadCloud, X, HelpCircle, Check, Sparkles } from 'lucide-react'
import Button from '../ui/Button'
import CameraCapture from './CameraCapture'
import { SpeechToTextButton, LanguageSelectorDropdown } from './VoiceInteractionBar'
import { MAX_IMAGE_SIZE_MB } from '../../constants'

const QUERY_MAX_HEIGHT = 160

/**
 * Upload & Camera Capture Area with drag-and-drop, direct camera mode,
 * live leaf preview, heatmap overlay toggle, image removal, context input, and vision engine selector.
 */
export default function UploadCameraArea({
  image,
  previewUrl,
  heatmapUrl,
  query,
  engine = 'hybrid',
  language = 'en',
  status,
  onSelectImage,
  onClearImage,
  onQueryChange,
  onEngineChange,
  onLanguageChange,
  onAnalyze,
  disabled = false,
}) {
  const [mode, setMode] = useState('upload') // 'upload' | 'camera'
  const [isDragging, setIsDragging] = useState(false)
  const [previewViewMode, setPreviewViewMode] = useState('original') // 'original' | 'heatmap'
  const [previewOverlayOpacity, setPreviewOverlayOpacity] = useState(80) // 0 to 100
  const fileInputRef = useRef(null)
  const cameraInputRef = useRef(null)
  const queryRef = useRef(null)

  const handleCameraClick = () => {
    if (disabled) return
    // In non-secure contexts (e.g. HTTP on local network/phone), getUserMedia is blocked by browser security policy.
    // Triggering native HTML5 capture="environment" opens the phone camera directly on any origin.
    if (!navigator.mediaDevices?.getUserMedia) {
      cameraInputRef.current?.click()
    } else {
      setMode('camera')
    }
  }

  useEffect(() => {
    const el = queryRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, QUERY_MAX_HEIGHT)}px`
  }, [query])

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (!disabled) setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    if (disabled) return
    const droppedFile = e.dataTransfer?.files?.[0]
    if (droppedFile) {
      onSelectImage(droppedFile)
    }
  }

  const handleFileInputChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      onSelectImage(file)
    }
  }

  const formatFileSize = (bytes) => {
    if (!bytes) return ''
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="space-y-4">
      {/* Vision Inference Engine Selector */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 rounded-xl border border-border-light bg-slate-900/[0.02] p-3 dark:border-border dark:bg-white/[0.02]">
        <div className="flex items-center gap-2">
          <Cpu size={15} className="text-accent-500 shrink-0" />
          <span className="text-xs font-semibold text-slate-700 dark:text-ink-secondary">Inference Engine:</span>
        </div>
        <div className="flex flex-wrap items-center justify-center gap-1.5">
          <button
            type="button"
            onClick={() => onEngineChange?.('hybrid')}
            disabled={disabled || status === 'analyzing' || status === 'streaming'}
            className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-all ${
              engine === 'hybrid'
                ? 'bg-accent-600 text-white shadow-sm'
                : 'bg-slate-900/5 text-slate-600 hover:bg-slate-900/10 dark:bg-white/5 dark:text-ink-muted dark:hover:bg-white/10'
            }`}
          >
            Hybrid Arbiter (Recommended)
          </button>
          <button
            type="button"
            onClick={() => onEngineChange?.('leafsense')}
            disabled={disabled || status === 'analyzing' || status === 'streaming'}
            className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-all ${
              engine === 'leafsense'
                ? 'bg-accent-600 text-white shadow-sm'
                : 'bg-slate-900/5 text-slate-600 hover:bg-slate-900/10 dark:bg-white/5 dark:text-ink-muted dark:hover:bg-white/10'
            }`}
          >
            Custom Model (LeafSense Port 8001)
          </button>
          <button
            type="button"
            onClick={() => onEngineChange?.('gemini')}
            disabled={disabled || status === 'analyzing' || status === 'streaming'}
            className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-all ${
              engine === 'gemini'
                ? 'bg-accent-600 text-white shadow-sm'
                : 'bg-slate-900/5 text-slate-600 hover:bg-slate-900/10 dark:bg-white/5 dark:text-ink-muted dark:hover:bg-white/10'
            }`}
          >
            Multimodal Vision (Gemini)
          </button>
        </div>
      </div>

      {/* Main Upload / Camera Box */}
      <AnimatePresence>
        {!previewUrl ? (
          mode === 'camera' ? (
            <motion.div
              key="camera-mode"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-3"
            >
              <CameraCapture
                onCapture={(file) => {
                  onSelectImage(file)
                }}
                onCancel={() => setMode('upload')}
                disabled={disabled}
              />
            </motion.div>
          ) : (
            <motion.div
              key="upload-dropzone"
              data-testid="leaf-dropzone"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`panel relative flex flex-col items-center justify-center rounded-panel border-2 border-dashed px-6 py-12 text-center transition-all ${
                isDragging
                  ? 'border-accent-500 bg-accent-500/10 dark:border-accent-400 dark:bg-accent-500/15'
                  : 'border-border-light hover:border-slate-300 dark:border-border dark:hover:border-white/20'
              }`}
            >
              <span className="flex h-14 w-14 items-center justify-center rounded-2xl border border-border-light bg-slate-900/5 text-slate-600 dark:border-border dark:bg-white/[0.04] dark:text-ink-secondary">
                <Leaf size={28} strokeWidth={1.5} className="text-accent-600 dark:text-accent-400" />
              </span>

              <div className="mt-3 space-y-1">
                <p className="font-display text-base font-semibold text-slate-800 dark:text-ink-primary">
                  {isDragging ? 'Drop your leaf photo here' : 'Select or Snap a Plant Leaf Photo'}
                </p>
                <p className="text-xs text-slate-500 dark:text-ink-muted">
                  Choose a saved photo from your device or take a live picture with your camera. Supports JPG, PNG, WEBP up to {MAX_IMAGE_SIZE_MB}MB.
                </p>
              </div>

              <div className="mt-5 flex flex-wrap items-center justify-center gap-3">
                <Button
                  type="button"
                  variant="primary"
                  icon={ImagePlus}
                  onClick={() => fileInputRef.current?.click()}
                  disabled={disabled}
                  aria-label="Upload leaf photo"
                >
                  Choose from Gallery / Files
                </Button>
                <Button
                  type="button"
                  variant="secondary"
                  icon={Camera}
                  onClick={handleCameraClick}
                  disabled={disabled}
                  aria-label="Take leaf photo"
                >
                  Take Photo with Camera
                </Button>
              </div>

              <input
                ref={fileInputRef}
                data-testid="leaf-file-input"
                type="file"
                accept="image/*"
                className="hidden"
                disabled={disabled}
                onChange={handleFileInputChange}
              />
              <input
                ref={cameraInputRef}
                data-testid="leaf-camera-input"
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                disabled={disabled}
                onChange={handleFileInputChange}
              />
            </motion.div>
          )
        ) : (
          /* Live Image Preview Mode */
          <motion.div
            key="preview-box"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            <div className="panel relative overflow-hidden rounded-panel border border-border-light dark:border-border">
              {/* Top metadata toolbar */}
              <div className="flex items-center justify-between border-b border-border-light bg-slate-900/[0.03] px-4 py-2.5 dark:border-border dark:bg-white/[0.02]">
                <div className="flex items-center gap-2 truncate text-xs text-slate-600 dark:text-ink-secondary">
                  <Leaf size={14} className="shrink-0 text-accent-500" />
                  <span className="truncate font-medium">{image?.name || 'Selected leaf image'}</span>
                  {image?.size && (
                    <span className="shrink-0 text-slate-400 dark:text-ink-muted">
                      ({formatFileSize(image.size)})
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={onClearImage}
                    disabled={status === 'analyzing'}
                    className="flex h-7 w-7 items-center justify-center rounded-lg text-slate-400 transition-colors hover:bg-rose-500/10 hover:text-rose-600 disabled:opacity-50 dark:hover:text-rose-400"
                    aria-label="Remove image"
                    title="Remove image"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>

              {/* Interactive Visual Toggle & Opacity Slider Toolbar (when heatmap is present) */}
              {heatmapUrl && (
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border-light bg-slate-900/[0.02] px-4 py-2 text-xs dark:border-border dark:bg-white/[0.01]">
                  <div className="flex items-center gap-1.5 rounded-lg border border-border-light bg-slate-900/5 p-0.5 dark:border-border dark:bg-white/5">
                    <button
                      type="button"
                      data-testid="upload-toggle-original-photo"
                      onClick={() => setPreviewViewMode('original')}
                      className={`rounded px-2 py-0.5 text-xs font-medium transition-all ${
                        previewViewMode === 'original'
                          ? 'bg-white text-slate-900 shadow-sm dark:bg-accent-600 dark:text-white'
                          : 'text-slate-600 hover:text-slate-900 dark:text-ink-muted dark:hover:text-white'
                      }`}
                    >
                      Original Photo
                    </button>
                    <button
                      type="button"
                      data-testid="upload-toggle-heatmap-overlay"
                      onClick={() => setPreviewViewMode('heatmap')}
                      className={`rounded px-2 py-0.5 text-xs font-medium transition-all ${
                        previewViewMode === 'heatmap'
                          ? 'bg-white text-slate-900 shadow-sm dark:bg-accent-600 dark:text-white'
                          : 'text-slate-600 hover:text-slate-900 dark:text-ink-muted dark:hover:text-white'
                      }`}
                    >
                      Heatmap Lesion Overlay
                    </button>
                  </div>

                  <div className="flex items-center gap-2">
                    <Sliders size={13} className="text-accent-500 shrink-0" />
                    <label htmlFor="upload-heatmap-opacity-slider" className="text-[11px] text-slate-500 dark:text-ink-muted">
                      Opacity:
                    </label>
                    <input
                      id="upload-heatmap-opacity-slider"
                      data-testid="upload-heatmap-opacity-slider"
                      type="range"
                      min="0"
                      max="100"
                      value={previewOverlayOpacity}
                      onChange={(e) => setPreviewOverlayOpacity(Number(e.target.value))}
                      disabled={previewViewMode === 'original'}
                      className="h-1.5 w-20 cursor-pointer appearance-none rounded-lg bg-slate-200 accent-accent-500 disabled:opacity-40 dark:bg-slate-700"
                    />
                    <span className="w-7 font-mono text-[10px] text-slate-500 dark:text-ink-muted">
                      {previewViewMode === 'original' ? '0%' : `${previewOverlayOpacity}%`}
                    </span>
                  </div>
                </div>
              )}

              {/* Image Preview Viewport */}
              <div className="relative flex max-h-80 items-center justify-center bg-slate-950/40 p-2">
                <img
                  src={previewUrl}
                  alt="Plant leaf preview"
                  className={`max-h-72 w-full rounded-lg object-contain transition-opacity ${
                    status === 'analyzing' ? 'opacity-50' : 'opacity-100'
                  }`}
                />

                {heatmapUrl && (
                  <img
                    src={heatmapUrl}
                    alt="Plant leaf heatmap overlay"
                    style={{
                      opacity: previewViewMode === 'original' ? 0 : previewOverlayOpacity / 100,
                    }}
                    className="pointer-events-none absolute inset-0 m-auto max-h-72 w-full rounded-lg object-contain transition-opacity duration-200"
                  />
                )}

                {status === 'analyzing' && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/60 text-white backdrop-blur-[2px]">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
                      className="flex h-12 w-12 items-center justify-center rounded-xl bg-accent-500/20 text-accent-400 ring-2 ring-accent-500/40"
                    >
                      <ScanLine size={24} />
                    </motion.div>
                    <p className="mt-3 text-sm font-medium">Scanning leaf pathology...</p>
                    <p className="text-xs text-slate-300">LeafSense Vision & RAG Treatment Engine</p>
                  </div>
                )}
              </div>
            </div>

            {/* Context / Follow-up Query Textarea */}
            <div className="panel rounded-panel p-4">
              <div className="mb-1.5 flex items-center justify-between">
                <label
                  htmlFor="diagnose-query-input"
                  className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-ink-secondary"
                >
                  <span>Optional context or symptoms observed</span>
                  <span className="hidden text-[11px] text-slate-400 sm:inline dark:text-ink-muted">e.g. &quot;Appeared after 3 days of rain&quot;</span>
                </label>
                <div className="flex items-center gap-2">
                  <LanguageSelectorDropdown
                    language={language}
                    onChange={onLanguageChange}
                    disabled={status === 'analyzing' || disabled}
                  />
                  <SpeechToTextButton
                    size="sm"
                    language={language}
                    disabled={status === 'analyzing' || disabled}
                    onTranscript={(spokenText) => {
                      const updated = query ? `${query} ${spokenText}` : spokenText
                      onQueryChange?.(updated)
                    }}
                  />
                </div>
              </div>
              <textarea
                ref={queryRef}
                id="diagnose-query-input"
                rows={2}
                value={query}
                onChange={(e) => onQueryChange(e.target.value)}
                disabled={status === 'analyzing'}
                placeholder="Add any specific context, crop stage, or question (or use mic)..."
                className="max-h-[160px] w-full resize-none rounded-lg border border-border-light bg-white/60 px-3 py-2 text-sm text-slate-800 placeholder-slate-400 outline-none transition-colors focus:border-accent-500 focus:ring-2 focus:ring-accent-500/30 dark:border-border dark:bg-white/[0.03] dark:text-ink-primary dark:placeholder-ink-muted"
              />
            </div>

            {/* Action Bar */}
            <div className="flex flex-wrap items-center justify-center gap-3">
              <Button
                type="button"
                variant="ghost"
                icon={RefreshCw}
                onClick={onClearImage}
                disabled={status === 'analyzing'}
              >
                Choose Another Photo
              </Button>
              <Button
                type="button"
                variant="primary"
                icon={ScanLine}
                loading={status === 'analyzing'}
                onClick={onAnalyze}
                disabled={status === 'analyzing' || disabled}
                className="min-w-[160px]"
              >
                {status === 'analyzing' ? 'Analyzing Leaf...' : 'Diagnose Plant Leaf'}
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
