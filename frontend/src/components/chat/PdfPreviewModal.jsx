import { useEffect, useRef, useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import { AlertCircle, Loader2 } from 'lucide-react'
import Modal from '../ui/Modal'
import api from '../../services/api'

// One worker for the whole app — pdfjs requires a single global worker
// source; subsequent pages reuse it.
pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl

/**
 * Renders one page of a document's source PDF (fetched from
 * GET /documents/{id}/file) with the cited excerpt highlighted via
 * GET /documents/{id}/pages/{page}/highlight?text=... The highlight rects
 * arrive in PDF points and are scaled onto the rendered canvas's actual
 * pixel size.
 */
export default function PdfPreviewModal({ documentId, pageNumber, excerpt, open, onClose, title }) {
  const [pdfUrl, setPdfUrl] = useState(null)
  const [highlightRects, setHighlightRects] = useState([])
  const [error, setError] = useState(null)
  const canvasRef = useRef(null)
  const scaleRef = useRef(1.5)

  useEffect(() => {
    if (!open || !documentId) return undefined

    let active = true
    setError(null)
    setPdfUrl(null)
    setHighlightRects([])

    const load = async () => {
      try {
        const [fileRes, highlightRes] = await Promise.all([
          api.get(`/documents/${encodeURIComponent(documentId)}/file`, { responseType: 'blob' }),
          excerpt
            ? api.get(
                `/documents/${encodeURIComponent(documentId)}/pages/${pageNumber}/highlight`,
                { params: { text: excerpt.slice(0, 300) } }
              )
            : Promise.resolve({ data: { rects: [] } }),
        ])
        if (!active) return
        setPdfUrl(URL.createObjectURL(fileRes.data))
        setHighlightRects(highlightRes.data?.rects ?? [])
      } catch (err) {
        if (active) setError(err?.response?.data?.detail ?? 'Could not load the document.')
      }
    }
    load()

    return () => {
      active = false
    }
  }, [open, documentId, pageNumber, excerpt])

  useEffect(() => {
    if (!open || !pdfUrl) return undefined
    let active = true

    const render = async () => {
      try {
        const pdf = await pdfjsLib.getDocument({ url: pdfUrl, disableWorker: false }).promise
        if (!active) return
        const clampedPage = Math.min(Math.max(1, pageNumber), pdf.numPages)
        const page = await pdf.getPage(clampedPage)
        const viewport = page.getViewport({ scale: scaleRef.current })
        const canvas = canvasRef.current
        if (!canvas) return
        canvas.width = viewport.width
        canvas.height = viewport.height
        const context = canvas.getContext('2d')
        await page.render({ canvasContext: context, viewport }).promise

        // Overlay highlight rects (in PDF points) scaled to canvas pixels.
        const overlay = canvas.parentElement?.querySelector('.pdf-highlight-overlay')
        if (overlay && highlightRects.length > 0) {
          overlay.style.width = `${viewport.width}px`
          overlay.style.height = `${viewport.height}px`
          overlay.innerHTML = ''
          for (const [x0, y0, x1, y1] of highlightRects) {
            const rect = document.createElement('div')
            rect.style.position = 'absolute'
            rect.style.left = `${x0 * scaleRef.current}px`
            rect.style.top = `${y0 * scaleRef.current}px`
            rect.style.width = `${(x1 - x0) * scaleRef.current}px`
            rect.style.height = `${(y1 - y0) * scaleRef.current}px`
            rect.style.background = 'rgba(245, 158, 11, 0.45)'
            rect.style.mixBlendMode = 'multiply'
            overlay.appendChild(rect)
          }
        }
      } catch (err) {
        if (active) setError('Could not render the document page.')
      }
    }
    render()

    return () => {
      active = false
    }
  }, [open, pdfUrl, pageNumber, highlightRects])

  useEffect(() => {
    if (!open) return undefined
    return () => {
      if (pdfUrl) URL.revokeObjectURL(pdfUrl)
    }
  }, [open, pdfUrl])

  return (
    <Modal open={open} onClose={onClose} title={title ?? 'Source document'} size="full">
      <div className="flex max-h-[70vh] flex-col items-center justify-center overflow-auto rounded-xl bg-slate-950/5 dark:bg-black/30">
        {error ? (
          <p className="flex items-center gap-2 p-8 text-sm text-danger">
            <AlertCircle size={15} /> {error}
          </p>
        ) : !pdfUrl ? (
          <p className="flex items-center gap-2 p-8 text-sm text-slate-500 dark:text-ink-muted">
            <Loader2 size={15} className="animate-spin" /> Loading page {pageNumber}…
          </p>
        ) : (
          <div className="relative">
            <canvas ref={canvasRef} className="max-w-full shadow-soft" />
            <div className="pdf-highlight-overlay pointer-events-none absolute inset-0" />
          </div>
        )}
      </div>
    </Modal>
  )
}
