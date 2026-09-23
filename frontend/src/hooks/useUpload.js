import { useCallback, useEffect, useRef, useState } from 'react'
import { addToUploadHistory, uploadDocument } from '../services/documentService'
import getErrorMessage from '../utils/errorMessage'
import useToast from './useToast'
import { MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB } from '../constants'

export default function useUpload() {
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('idle') // idle | uploading | success | error
  const [errorMessage, setErrorMessage] = useState('')
  const { showToast } = useToast()
  // Aborts an in-flight upload if the component unmounts (e.g. the user
  // navigates away during a long upload) so the request doesn't resolve into
  // a toast/hook callback on an unmounted context. Reset on every new upload.
  const abortControllerRef = useRef(null)

  const selectFile = useCallback(
    (selected) => {
      if (selected.type !== 'application/pdf') {
        showToast('Please choose a PDF file.', 'error')
        return
      }
      if (selected.size > MAX_UPLOAD_SIZE_BYTES) {
        showToast(`PDF exceeds the ${MAX_UPLOAD_SIZE_MB} MB size limit.`, 'error')
        return
      }
      setFile(selected)
      setStatus('idle')
      setProgress(0)
      setErrorMessage('')
    },
    [showToast]
  )

  const reset = useCallback(() => {
    setFile(null)
    setStatus('idle')
    setProgress(0)
    setErrorMessage('')
  }, [])

  const startUpload = useCallback(
    async (collection) => {
      if (!file) return
      setStatus('uploading')
      setProgress(0)
      setErrorMessage('')
       abortControllerRef.current?.abort()
       const controller = new AbortController()
       abortControllerRef.current = controller
       const signal = controller.signal

      try {
        const result = await uploadDocument(file, setProgress, collection, signal)
        setStatus('success')
        // DocumentProcessingResponse carries no timestamp — stamp one
        // locally so Documents can sort/display by upload time.
        addToUploadHistory({ ...result, uploaded_at: new Date().toISOString() })
        // pages_ocred > 0 means some pages had no text layer (scanned/image
        // pages) and were recovered via OCR — worth surfacing, since OCR'd
        // text is lower-fidelity than the document's real text. Same
        // reasoning for the multi-modal counts below (all 0 when those
        // features are off, so this is a no-op note on a normal deployment).
        const notes = []
        if (result.pages_ocred > 0) notes.push(`${result.pages_ocred} page(s) recovered via OCR`)
        if (result.images_captioned > 0) notes.push(`${result.images_captioned} image(s) captioned`)
        if (result.total_tables > 0) notes.push(`${result.total_tables} table(s) extracted`)
        if (result.possible_duplicate_of) notes.push('may duplicate an already-uploaded document')
        const note = notes.length > 0 ? ` (${notes.join(', ')})` : ''
        showToast(`${result.original_filename} uploaded successfully${note}.`, 'success')
      } catch (error) {
        // An aborted upload isn't a failure — it was cancelled deliberately
        // (new upload starting, or component unmounting). Don't flash an
        // error toast or leave the hook in an error state for it.
        if (signal.aborted) return
        const message = getErrorMessage(error, 'Upload failed. Please try again.')
        setStatus('error')
        setErrorMessage(message)
        showToast(message, 'error')
      } finally {
        // Only clear if no newer upload has replaced this controller — the
        // finally of a deliberately-cancelled (aborted) upload must not null
        // out a still-in-flight newer one and defeat its unmount abort.
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null
        }
      }
    },
    [file, showToast]
  )

  // Abort any in-flight upload on unmount so a request can't resolve into a
  // toast after the component is gone. The catch above swallows the abort
  // quietly, leaving whatever status was already committed (idle/uploading).
  useEffect(() => () => abortControllerRef.current?.abort(), [])

  return { file, progress, status, errorMessage, selectFile, reset, startUpload }
}
