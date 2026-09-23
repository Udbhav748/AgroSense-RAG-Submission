import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Camera, CameraOff, ImagePlus, RefreshCw } from 'lucide-react'
import Button from '../ui/Button'

/**
 * Live camera capture with a file-upload fallback.
 *
 * On mount it requests a stream from getUserMedia (back camera preferred for
 * a leaf photo). When the user presses "Capture", the current video frame is
 * drawn to a hidden canvas and converted to a File via toBlob. If the camera
 * is unavailable or permission is denied, a file picker fallback is shown so
 * the feature still works from a phone's existing photos.
 */
export default function CameraCapture({ onCapture, onCancel, disabled }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const facingModeRef = useRef('environment')
  const fileInputRef = useRef(null)
  const nativeCameraInputRef = useRef(null)
  const [cameraState, setCameraState] = useState('starting') // starting | active | error
  const [cameraError, setCameraError] = useState('')
  const [switching, setSwitching] = useState(false)

  const stopStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
  }

  useEffect(() => {
    let cancelled = false

    async function startCamera() {
      setCameraState('starting')
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraError('Camera is not supported in this browser. You can upload a photo instead.')
        setCameraState('error')
        return
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop())
          return
        }
        streamRef.current = stream
        facingModeRef.current = 'environment'
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        setCameraError('')
        setCameraState('active')
      } catch (error) {
        if (cancelled) return
        const message =
          error?.name === 'NotAllowedError'
            ? 'Camera permission was denied. Allow camera access, or upload a photo instead.'
            : 'Could not start the camera. You can upload a photo instead.'
        setCameraError(message)
        setCameraState('error')
      }
    }

    startCamera()
    return () => {
      cancelled = true
      stopStream()
    }
  }, [])

  const capture = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    canvas.width = video.videoWidth || 1280
    canvas.height = video.videoHeight || 720
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92))
    if (!blob) return
    const file = new File([blob], `leaf-capture-${Date.now()}.jpg`, { type: 'image/jpeg' })
    onCapture(file)
  }

  const switchCamera = async () => {
    setSwitching(true)
    stopStream()
    const nextMode = facingModeRef.current === 'environment' ? 'user' : 'environment'
    facingModeRef.current = nextMode
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: nextMode },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setCameraState('active')
    } catch {
      facingModeRef.current = 'environment'
      setCameraError('Could not switch camera. You can upload a photo instead.')
      setCameraState('error')
    } finally {
      setSwitching(false)
    }
  }

  return (
    <div className="space-y-4">
      {onCancel && (
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={onCancel}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 dark:text-ink-muted dark:hover:text-ink-primary"
          >
            ← Back to File Upload
          </button>
        </div>
      )}

      <div className="panel relative overflow-hidden rounded-panel bg-slate-950">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`h-72 w-full object-cover ${cameraState === 'active' ? '' : 'hidden'}`}
        />
        {cameraState !== 'active' && (
          <div className="flex h-72 flex-col items-center justify-center gap-3 px-6 text-center">
            <span className="flex h-12 w-12 items-center justify-center rounded-xl border border-border-light bg-slate-900/5 text-slate-600 dark:border-border dark:bg-white/5 dark:text-ink-muted">
              <Camera size={22} strokeWidth={1.75} className="text-accent-500" />
            </span>
            <div className="space-y-1">
              <p className="font-medium text-sm text-slate-800 dark:text-ink-primary">
                {cameraState === 'starting' ? 'Initializing camera...' : 'Use Native Phone Camera'}
              </p>
              <p className="max-w-sm text-xs text-slate-500 dark:text-ink-muted">
                {cameraError || 'Tap below to snap a high-resolution photo with your device camera.'}
              </p>
            </div>
            {cameraState === 'error' && (
              <div className="mt-2 flex flex-wrap items-center justify-center gap-2">
                <Button
                  type="button"
                  variant="primary"
                  icon={Camera}
                  onClick={() => nativeCameraInputRef.current?.click()}
                  disabled={disabled}
                >
                  Open Device Camera
                </Button>
                {onCancel ? (
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={onCancel}
                    disabled={disabled}
                  >
                    Back to File Upload
                  </Button>
                ) : (
                  <Button
                    type="button"
                    variant="secondary"
                    icon={ImagePlus}
                    onClick={() => fileInputRef.current?.click()}
                    disabled={disabled}
                  >
                    Choose from Gallery
                  </Button>
                )}
              </div>
            )}
          </div>
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {cameraState === 'active' && (
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Button
            variant="primary"
            icon={Camera}
            disabled={disabled}
            onClick={capture}
          >
            Capture Photo
          </Button>
          <Button
            variant="secondary"
            icon={RefreshCw}
            disabled={disabled || switching}
            loading={switching}
            onClick={switchCamera}
          >
            Switch Camera
          </Button>
          {onCancel && (
            <Button
              variant="ghost"
              disabled={disabled}
              onClick={onCancel}
            >
              Back to Upload
            </Button>
          )}
        </div>
      )}

      {/* Native Mobile Camera Input */}
      <input
        ref={nativeCameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        disabled={disabled}
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onCapture(file)
          event.target.value = ''
        }}
      />

      {/* Standard File Picker Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        disabled={disabled}
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onCapture(file)
          event.target.value = ''
        }}
      />

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-center text-xs text-slate-400 dark:text-ink-muted"
      >
        Hold the leaf flat, in good light, filling most of the frame.
      </motion.p>
    </div>
  )
}
