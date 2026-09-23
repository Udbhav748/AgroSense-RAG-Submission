import { useEffect, useState } from 'react'
import { AlertCircle, ImageOff, Loader2 } from 'lucide-react'
import Modal from '../ui/Modal'
import { fetchDocumentImageBlob, listDocumentImages } from '../../services/documentService'

// content_type here is DocumentImageItem's ("figure" | "page"), a
// different vocabulary from SourceReference's content_type
// ("image_caption" | "table" | "text") used elsewhere in the chat UI —
// this one describes what LeafSense's extraction pulled out, not what a
// citation was derived from.
function imageLabel(image) {
  return image.content_type === 'page' ? `Page ${image.page_number} (full render)` : `Figure — page ${image.page_number}`
}

/**
 * Gallery of a document's extracted images (multi-modal RAG). Each
 * thumbnail is fetched as an authenticated blob (see
 * fetchDocumentImageBlob's docstring for why a plain <img src> can't be
 * used here) and revoked on close/unmount so this doesn't leak object
 * URLs across repeated opens.
 */
export default function ImageGalleryModal({ documentId, title, open, onClose }) {
  const [images, setImages] = useState([])
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [errorMessage, setErrorMessage] = useState('')

  useEffect(() => {
    if (!open || !documentId) return undefined

    let active = true
    let objectUrls = []
    setStatus('loading')
    setImages([])

    const load = async () => {
      try {
        const items = await listDocumentImages(documentId)
        const withBlobs = await Promise.all(
          items.map(async (item) => {
            const objectUrl = await fetchDocumentImageBlob(item.url)
            return { ...item, objectUrl }
          })
        )
        if (!active) {
          withBlobs.forEach((item) => URL.revokeObjectURL(item.objectUrl))
          return
        }
        objectUrls = withBlobs.map((item) => item.objectUrl)
        setImages(withBlobs)
        setStatus('success')
      } catch (err) {
        if (active) {
          setErrorMessage(err?.response?.data?.detail ?? 'Could not load this document\'s images.')
          setStatus('error')
        }
      }
    }
    load()

    return () => {
      active = false
      objectUrls.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [open, documentId])

  return (
    <Modal open={open} onClose={onClose} title={title ?? 'Extracted images'} size="full">
      <div className="max-h-[70vh] overflow-y-auto">
        {status === 'loading' && (
          <p className="flex items-center justify-center gap-2 p-10 text-sm text-slate-500 dark:text-ink-muted">
            <Loader2 size={15} className="animate-spin" /> Loading images…
          </p>
        )}
        {status === 'error' && (
          <p className="flex items-center justify-center gap-2 p-10 text-sm text-danger">
            <AlertCircle size={15} /> {errorMessage}
          </p>
        )}
        {status === 'success' && images.length === 0 && (
          <p className="flex flex-col items-center justify-center gap-2 p-10 text-center text-sm text-slate-500 dark:text-ink-muted">
            <ImageOff size={20} />
            No images were extracted from this document.
          </p>
        )}
        {status === 'success' && images.length > 0 && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
            {images.map((image) => (
              <div
                key={image.image_id}
                className="overflow-hidden rounded-lg border border-border-light dark:border-border"
              >
                <img
                  src={image.objectUrl}
                  alt={imageLabel(image)}
                  className="max-h-56 w-full bg-slate-950/5 object-contain dark:bg-black/20"
                />
                <p className="px-2.5 py-1.5 text-xs text-slate-500 dark:text-ink-muted">{imageLabel(image)}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </Modal>
  )
}
