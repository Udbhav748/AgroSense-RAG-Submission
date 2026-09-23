import { useCallback, useEffect, useRef, useState } from 'react'
import { checkLeafSenseHealth, diagnoseImageStream, diagnoseLeaf } from '../services/diagnoseService'
import getErrorMessage, { getErrorInfo } from '../utils/errorMessage'
import useToast from './useToast'
import { MAX_IMAGE_SIZE_BYTES, MAX_IMAGE_SIZE_MB } from '../constants'

export default function useDiagnose() {
  const [image, setImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [query, setQuery] = useState('')
  const [engine, setEngine] = useState('hybrid') // hybrid | leafsense | gemini
  const [language, setLanguage] = useState('en')
  const [status, setStatus] = useState('idle') // idle | preview | analyzing | streaming | success | error
  const [result, setResult] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [isLeafSenseOnline, setIsLeafSenseOnline] = useState(true)
  const [isCheckingHealth, setIsCheckingHealth] = useState(false)
  const previewUrlRef = useRef(null)
  const { showToast } = useToast()

  const revokePreview = useCallback(() => {
    if (previewUrlRef.current) {
      try {
        URL.revokeObjectURL(previewUrlRef.current)
      } catch {
        // Ignored in environments where revokeObjectURL is unsupported
      }
      previewUrlRef.current = null
    }
  }, [])

  useEffect(() => revokePreview, [revokePreview])

  // Periodic or on-demand health check for LeafSense port 8001
  const checkHealth = useCallback(async () => {
    setIsCheckingHealth(true)
    try {
      const res = await checkLeafSenseHealth()
      setIsLeafSenseOnline(res.online)
      return res.online
    } catch {
      setIsLeafSenseOnline(false)
      return false
    } finally {
      setIsCheckingHealth(false)
    }
  }, [])

  // Auto-poll health: every 3s if offline (to catch startup instantly), every 20s if online
  useEffect(() => {
    checkHealth()
    const intervalMs = isLeafSenseOnline ? 20000 : 3000
    const interval = setInterval(checkHealth, intervalMs)
    return () => clearInterval(interval)
  }, [checkHealth, isLeafSenseOnline])

  const selectImage = useCallback(
    (file) => {
      if (!file) return
      const isImage =
        (file.type || '').startsWith('image/') ||
        /\.(jpe?g|png|webp|gif|bmp|tiff?)$/i.test(file.name || '')

      if (!isImage) {
        showToast('Please choose an image file (JPG, PNG, WEBP, ...).', 'error')
        return
      }
      if (file.size > MAX_IMAGE_SIZE_BYTES) {
        showToast(`Image exceeds the ${MAX_IMAGE_SIZE_MB} MB size limit.`, 'error')
        return
      }
      revokePreview()
      let url = ''
      try {
        url = URL.createObjectURL(file)
      } catch {
        url = 'blob:http://localhost/mock-leaf-preview'
      }
      previewUrlRef.current = url
      setImage(file)
      setPreviewUrl(url)
      setResult(null)
      setErrorMessage('')
      setStatus('preview')
    },
    [revokePreview, showToast]
  )

  const clearImage = useCallback(() => {
    revokePreview()
    setImage(null)
    setPreviewUrl(null)
    if (status === 'preview') {
      setStatus('idle')
    }
  }, [revokePreview, status])

  const setQueryText = useCallback((text) => setQuery(text), [])

  const analyze = useCallback(async () => {
    if (!image || status === 'analyzing' || status === 'streaming') return
    setStatus('analyzing')
    setErrorMessage('')
    setResult(null)

    let currentResult = {
      diagnosis: null,
      answer: '',
      sources: [],
      processing_time: null,
      session_id: null,
      tool_used: 'diagnose',
      steps_taken: 1,
      retrieved_chunks: [],
    }

    try {
      await diagnoseImageStream(
        image,
        { query, engine, language },
        (event) => {
          if (event.type === 'diagnosis') {
            const diagData = event.diagnosis || event.payload || event
            currentResult = {
              ...currentResult,
              diagnosis: diagData,
            }
            setResult({ ...currentResult })
            setStatus('streaming')
            setIsLeafSenseOnline(true)
          } else if (event.type === 'answer_chunk') {
            currentResult = {
              ...currentResult,
              answer: (currentResult.answer || '') + (event.text || ''),
            }
            setResult({ ...currentResult })
            setStatus('streaming')
          } else if (event.type === 'trace') {
            // Trace telemetry received
          } else if (event.type === 'done') {
            const payload = event.payload || {}
            currentResult = {
              ...currentResult,
              ...payload,
              diagnosis: payload.diagnosis || currentResult.diagnosis,
              answer: payload.answer || currentResult.answer,
              sources: payload.sources || currentResult.sources || [],
              processing_time: payload.processing_time ?? currentResult.processing_time,
              session_id: payload.session_id || currentResult.session_id,
            }
            setResult({ ...currentResult })
            setStatus('success')
            setIsLeafSenseOnline(true)
          } else if (event.type === 'error') {
            const errDetail = event.detail || event
            const err = new Error(errDetail?.message || 'Diagnosis failed.')
            err.response = { data: errDetail }
            throw err
          }
        },
        (error) => {
          throw error
        }
      )
    } catch (error) {
      const errInfo = getErrorInfo(error)
      if (
        errInfo?.error_code === 'VISION_SERVICE_ERROR' ||
        error?.message?.includes('8001') ||
        error?.response?.data?.detail?.includes('8001')
      ) {
        setIsLeafSenseOnline(false)
      }
      const message = getErrorMessage(error, 'Diagnosis failed. Please try again.')
      setErrorMessage(message)
      setStatus('error')
      showToast(message, 'error')
    }
  }, [image, query, engine, language, status, showToast])

  const reset = useCallback(() => {
    revokePreview()
    setImage(null)
    setPreviewUrl(null)
    setQuery('')
    setResult(null)
    setErrorMessage('')
    setStatus('idle')
  }, [revokePreview])

  return {
    image,
    previewUrl,
    query,
    engine,
    setEngine,
    language,
    setLanguage,
    status,
    isStreaming: status === 'streaming',
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
  }
}

