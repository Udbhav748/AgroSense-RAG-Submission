import api from './api'

const HISTORY_KEY = 'insightai-upload-history'

/**
 * Upload a PDF to the backend, reporting real byte-level progress.
 * @param {File} file
 * @param {(percent: number) => void} [onProgress]
 * @param {string} [collection] optional named collection to tag the
 *   document into, later used to scope a chat conversation.
 * @param {AbortSignal} [signal] optional signal to abort the upload (the
 *   caller owns the AbortController; aborting is treated as a cancellation,
 *   not an error, by hooks/useUpload).
 * @returns {Promise<{document_id: string, original_filename: string, total_pages: number, total_chunks: number, total_embeddings: number, pages_ocred: number, processing_time: number, status: string}>}
 */
export async function uploadDocument(file, onProgress, collection, signal) {
  const formData = new FormData()
  formData.append('file', file)
  if (collection) formData.append('collection', collection)

  const { data } = await api.post('/upload', formData, {
    timeout: 60000,
    signal,
    onUploadProgress: (event) => {
      if (!onProgress || !event.total) return
      onProgress(Math.round((event.loaded / event.total) * 100))
    },
  })
  return data
}

/**
 * Delete a document from the server: removes its vectors from the FAISS
 * index and its uploaded file from disk. The backend requires explicit
 * confirmation (?confirm=true) or it rejects the request — callers must
 * only invoke this after the user has confirmed the deletion themselves
 * (see the confirm dialog in pages/Documents.jsx).
 * @param {string} documentId
 * @returns {Promise<{document_id: string, chunks_removed: number, status: string}>}
 */
export async function deleteDocument(documentId) {
  const { data } = await api.delete(`/documents/${encodeURIComponent(documentId)}`, {
    params: { confirm: true },
  })
  return data
}

/**
 * List documents from the server (GET /documents) — tenant-scoped to the
 * logged-in user, so this is what makes a document uploaded from a
 * different browser/device visible here too, unlike getUploadHistory()
 * below. Returns [] when the backend's DATABASE_URL isn't configured (the
 * route's own documented behavior: "empty when the DB is disabled") —
 * callers should treat that the same as "nothing from the server yet",
 * not as an error, and fall back to getUploadHistory() for that case.
 * `allTenants` is admin-only — the backend 403s it for any other role, so
 * callers must gate the UI on user.role === 'admin' before passing it.
 * @param {{ collection?: string, allTenants?: boolean }} [options]
 * @returns {Promise<{document_id: string, original_filename: string, total_pages: number, total_chunks: number, upload_timestamp: string, collection: string|null, tenant_id: number|null}[]>}
 */
export async function listDocuments(options = {}) {
  const params = {}
  if (options.collection) params.collection = options.collection
  if (options.allTenants) params.all_tenants = true
  const { data } = await api.get('/documents', { params: Object.keys(params).length ? params : undefined })
  return data.documents
}

/**
 * List images extracted from a document (GET /documents/{id}/images) —
 * multi-modal RAG (image_extraction_enabled), metadata only. Each item's
 * `url` is the API path to fetch the actual bytes from — authenticated,
 * so it can't be used as a plain <img src>; see fetchDocumentImageBlob.
 * @param {string} documentId
 * @returns {Promise<{image_id: string, document_id: string, page_number: number, content_type: string, mime_type: string, width: number, height: number, byte_size: number, url: string}[]>}
 */
export async function listDocumentImages(documentId) {
  const { data } = await api.get(`/documents/${encodeURIComponent(documentId)}/images`)
  return data.images
}

/**
 * Fetch one extracted image's bytes as an object URL — same
 * fetch-as-blob-then-objectURL pattern PdfPreviewModal.jsx uses for the
 * source PDF, needed because the image endpoint requires the same
 * Authorization header every other API call carries (api.js's
 * interceptor), which a plain <img src="..."> can't attach.
 * Caller owns the returned URL and must URL.revokeObjectURL it when done.
 * @param {string} url One image's `url` field from listDocumentImages.
 * @returns {Promise<string>}
 */
export async function fetchDocumentImageBlob(url) {
  const { data } = await api.get(url, { responseType: 'blob' })
  return URL.createObjectURL(data)
}

/**
 * The backend now has a real document list endpoint (listDocuments above),
 * but only when a database is configured — upload history is also tracked
 * client-side from real, successful uploads so Documents still works fully
 * on a DB-less deployment, and so documents predating server-side listing
 * aren't lost. This is real data about what this browser has uploaded,
 * not mock data.
 */
export function getUploadHistory() {
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export function addToUploadHistory(document) {
  const history = [document, ...getUploadHistory()]
  window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history))
  return history
}

export function removeFromUploadHistory(documentId) {
  const history = getUploadHistory().filter((doc) => doc.document_id !== documentId)
  window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history))
  return history
}

/**
 * The /chat API's retrieved_chunks only carry document_id, not a filename
 * (the backend never threads original_filename through chunking/embedding).
 * Resolve a friendly name from this browser's own upload history when the
 * document was uploaded here; otherwise return null so callers can fall
 * back to showing the id rather than a fabricated name.
 */
export function getDocumentName(documentId) {
  const match = getUploadHistory().find((doc) => doc.document_id === documentId)
  return match?.original_filename ?? null
}
