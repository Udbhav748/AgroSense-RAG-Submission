import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowUpDown, Eye, FileText, Image as ImageIcon, MessageSquare, ShieldCheck, Table2, Trash2, UploadCloud } from 'lucide-react'
import SearchInput from '../components/ui/SearchInput'
import StatusBadge from '../components/ui/StatusBadge'
import EmptyState from '../components/ui/EmptyState'
import Button from '../components/ui/Button'
import Modal from '../components/ui/Modal'
import PdfPreviewModal from '../components/chat/PdfPreviewModal'
import ImageGalleryModal from '../components/documents/ImageGalleryModal'
import Skeleton from '../components/ui/Skeleton'
import { deleteDocument, getUploadHistory, listDocuments, removeFromUploadHistory } from '../services/documentService'
import useToast from '../hooks/useToast'
import useAuth from '../hooks/useAuth'
import getErrorMessage from '../utils/errorMessage'

// listDocuments() (GET /documents) only carries the fields DocumentListItem
// declares — no status/images_captioned/total_tables, since those are
// upload-response-only fields the backend never persists. Local upload
// history has those richer fields for anything uploaded from this browser;
// merging keeps that richness where available while still surfacing
// documents the server knows about but this browser never uploaded itself
// (a different browser/device, or a document from before this feature
// existed at all).
function mergeDocuments(serverDocs, localDocs) {
  const localById = new Map(localDocs.map((doc) => [doc.document_id, doc]))
  const merged = serverDocs.map((doc) => {
    const normalized = { ...doc, uploaded_at: doc.upload_timestamp }
    const local = localById.get(doc.document_id)
    localById.delete(doc.document_id)
    return local ? { ...normalized, ...local } : normalized
  })
  // Whatever's left is local-only: either a DB-less deployment (the server
  // list is always [] there, so every document falls into this bucket) or
  // a document uploaded before server-side listing existed.
  return [...merged, ...localById.values()]
}

function formatDate(iso) {
  if (!iso) return 'Unknown date'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return 'Unknown date'
  return date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function effectiveStatus(doc) {
  if (doc.status === 'processed' && doc.total_chunks === 0) return 'warning'
  // doc.status only exists on the richer upload-response shape (merged in
  // from local history for anything this browser uploaded itself — see
  // mergeDocuments above). A doc with no local match came from GET
  // /documents alone, which only ever lists documents the server has
  // already fully indexed — defaulting that case to 'uploaded' read as
  // "still processing" for a document that's actually chat-ready.
  return doc.status ?? 'processed'
}

const SORT_OPTIONS = [
  { value: 'newest', label: 'Newest first' },
  { value: 'oldest', label: 'Oldest first' },
  { value: 'name', label: 'Name (A-Z)' },
]

export default function Documents() {
  const [documents, setDocuments] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [query, setQuery] = useState('')
  const [sortBy, setSortBy] = useState('newest')
  const [collectionFilter, setCollectionFilter] = useState('')
  const [pendingDelete, setPendingDelete] = useState(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const [previewDoc, setPreviewDoc] = useState(null)
  const [galleryDoc, setGalleryDoc] = useState(null)
  const [allTenants, setAllTenants] = useState(false)
  const navigate = useNavigate()
  const { showToast } = useToast()
  const { user } = useAuth()
  const isAdmin = user?.role === 'admin'

  useEffect(() => {
    let active = true
    setIsLoading(true)
    listDocuments({ allTenants: isAdmin && allTenants })
      .then((serverDocs) => {
        // Cross-tenant results aren't this browser's own uploads, so
        // merging in local history (keyed by document_id only) is still
        // safe — it can only enrich rows this browser genuinely uploaded.
        if (active) setDocuments(mergeDocuments(serverDocs, getUploadHistory()))
      })
      .catch(() => {
        // Server list unreachable (network error, auth hiccup, etc.) —
        // degrade to this browser's own history rather than show nothing.
        if (active) setDocuments(getUploadHistory())
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [isAdmin, allTenants])

  const collections = useMemo(
    () => [...new Set(documents.map((doc) => doc.collection).filter(Boolean))],
    [documents]
  )

  const filtered = useMemo(() => {
    const list = documents.filter((doc) => {
      const matchesQuery = doc.original_filename.toLowerCase().includes(query.toLowerCase())
      const matchesCollection = !collectionFilter || doc.collection === collectionFilter
      return matchesQuery && matchesCollection
    })
    const sorted = [...list].sort((a, b) => {
      if (sortBy === 'name') return a.original_filename.localeCompare(b.original_filename)
      const dateA = new Date(a.uploaded_at).getTime() || 0
      const dateB = new Date(b.uploaded_at).getTime() || 0
      return sortBy === 'oldest' ? dateA - dateB : dateB - dateA
    })
    return sorted
  }, [documents, query, sortBy, collectionFilter])

  const chatAboutCollection = () => {
    const matching = documents
      .filter((doc) => doc.collection === collectionFilter)
      .map((doc) => doc.document_id)
    navigate('/chat', { state: { documentIds: matching } })
  }

  const confirmDelete = async () => {
    if (!pendingDelete) return
    setIsDeleting(true)
    try {
      await deleteDocument(pendingDelete.document_id)
      removeFromUploadHistory(pendingDelete.document_id)
      setDocuments((current) => current.filter((doc) => doc.document_id !== pendingDelete.document_id))
      showToast(`${pendingDelete.original_filename} deleted.`, 'success')
      setPendingDelete(null)
    } catch (error) {
      showToast(getErrorMessage(error, 'Failed to delete document. Please try again.'), 'error')
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className="space-y-5 py-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="font-display text-xl font-bold">Documents</h2>
          <p className="text-sm text-slate-500 dark:text-ink-muted">
            {isLoading ? 'Loading…' : `${documents.length} document${documents.length === 1 ? '' : 's'}`}
            {isAdmin && allTenants && ' · all tenants'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isAdmin && (
            <button
              type="button"
              onClick={() => setAllTenants((current) => !current)}
              title="Admin: list documents across every tenant, not just your own"
              className={`flex h-10 shrink-0 items-center gap-1.5 rounded-xl border px-3.5 text-sm font-medium transition-colors ${
                allTenants
                  ? 'border-accent-500/50 bg-accent-500/10 text-accent-600 dark:text-accent-400'
                  : 'border-border-light text-slate-500 hover:bg-slate-900/5 dark:border-border dark:text-ink-muted dark:hover:bg-white/[0.04]'
              }`}
            >
              <ShieldCheck size={15} strokeWidth={1.75} />
              All tenants
            </button>
          )}
          <Button variant="primary" icon={UploadCloud} onClick={() => navigate('/upload')}>
            Upload
          </Button>
        </div>
      </div>

      {documents.length > 0 && (
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <SearchInput value={query} onChange={setQuery} placeholder="Search documents..." className="sm:w-72" />
          {collections.length > 0 && (
            <div className="relative">
              <select
                value={collectionFilter}
                onChange={(event) => setCollectionFilter(event.target.value)}
                className="input w-full appearance-none py-2 pl-8 pr-8 sm:w-44"
                aria-label="Filter by collection"
              >
                <option value="">All collections</option>
                {collections.map((collection) => (
                  <option key={collection} value={collection}>
                    {collection}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="relative sm:ml-auto">
            <ArrowUpDown size={13} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 dark:text-ink-muted" />
            <select
              value={sortBy}
              onChange={(event) => setSortBy(event.target.value)}
              className="input w-full appearance-none py-2 pl-8 pr-8 sm:w-44"
            >
              {SORT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          {collectionFilter && (
            <Button variant="secondary" icon={MessageSquare} onClick={chatAboutCollection}>
              Chat about this collection
            </Button>
          )}
        </div>
      )}

      {isLoading ? (
        <div className="panel divide-y divide-border-light overflow-hidden dark:divide-border">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center">
              <div className="flex min-w-0 flex-1 items-center gap-3">
                <Skeleton className="h-10 w-10 shrink-0 rounded-lg" />
                <div className="min-w-0 flex-1 space-y-2">
                  <Skeleton className="h-3.5 w-2/5" />
                  <Skeleton className="h-3 w-1/4" />
                </div>
              </div>
              <div className="flex items-center gap-4 pl-[3.25rem] sm:pl-0">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-5 w-16 rounded-full" />
              </div>
            </div>
          ))}
        </div>
      ) : documents.length === 0 ? (
        <EmptyState
          icon={FileText}
          title="No documents yet"
          description="Upload your first PDF to start building your knowledge base."
          action={
            <Button variant="primary" icon={UploadCloud} onClick={() => navigate('/upload')}>
              Upload a document
            </Button>
          }
        />
      ) : filtered.length === 0 ? (
        <EmptyState icon={FileText} title="No matches" description="Try a different search term." />
      ) : (
        <div className="panel divide-y divide-border-light overflow-hidden dark:divide-border">
          {filtered.map((doc, index) => (
            <motion.div
              key={doc.document_id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.25, delay: index * 0.03 }}
              className="flex flex-col gap-3 px-5 py-4 transition-colors hover:bg-slate-900/[0.02] dark:hover:bg-white/[0.02] sm:flex-row sm:items-center"
            >
              <div className="flex min-w-0 flex-1 items-center gap-3">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border-light bg-slate-900/5 text-slate-500 dark:border-border dark:bg-white/[0.03] dark:text-ink-secondary">
                  <FileText size={18} strokeWidth={1.5} />
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-slate-800 dark:text-ink-primary">
                    {doc.original_filename}
                  </p>
                  <p className="text-xs text-slate-400 dark:text-ink-muted">
                    {formatDate(doc.uploaded_at)}
                    {doc.collection && (
                      <span className="ml-1.5 rounded bg-accent-500/10 px-1.5 py-0.5 text-[10px] font-medium text-accent-600 dark:text-accent-400">
                        {doc.collection}
                      </span>
                    )}
                    {allTenants && doc.tenant_id != null && (
                      <span className="ml-1.5 rounded bg-slate-500/10 px-1.5 py-0.5 text-[10px] font-medium text-slate-500 dark:text-ink-muted">
                        Tenant #{doc.tenant_id}
                      </span>
                    )}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-4 pl-[3.25rem] sm:pl-0">
                <span className="w-28 shrink-0 text-xs text-slate-500 dark:text-ink-muted">
                  {doc.total_pages != null ? `${doc.total_pages} pg · ${doc.total_chunks} chunks` : '—'}
                </span>
                {/* Multi-modal extraction badges — only present once
                    images_captioned/total_tables exist on a stored upload
                    result, so this renders nothing for documents uploaded
                    before those fields existed or with the features off. */}
                {doc.images_captioned > 0 && (
                  <button
                    type="button"
                    onClick={() => setGalleryDoc(doc)}
                    title={`View ${doc.images_captioned} extracted image(s)`}
                    className="flex shrink-0 items-center gap-1 text-slate-400 transition-colors hover:text-accent-600 dark:text-ink-muted dark:hover:text-accent-500"
                  >
                    <ImageIcon size={13} strokeWidth={1.75} />
                  </button>
                )}
                {doc.total_tables > 0 && (
                  <span
                    title={`${doc.total_tables} table(s) extracted`}
                    className="flex shrink-0 items-center gap-1 text-slate-400 dark:text-ink-muted"
                  >
                    <Table2 size={13} strokeWidth={1.75} />
                  </span>
                )}
                <StatusBadge status={effectiveStatus(doc)} />
                <button
                  type="button"
                  onClick={() => setPreviewDoc(doc)}
                  aria-label={`View ${doc.original_filename}`}
                  className="ml-auto rounded-lg p-2 text-slate-400 transition-colors hover:bg-accent-500/10 hover:text-accent-600 dark:text-ink-muted dark:hover:text-accent-500 sm:ml-0"
                >
                  <Eye size={15} />
                </button>
                <button
                  type="button"
                  onClick={() => setPendingDelete(doc)}
                  aria-label={`Delete ${doc.original_filename}`}
                  className="rounded-lg p-2 text-slate-400 transition-colors hover:bg-danger/10 hover:text-danger dark:text-ink-muted"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      <Modal
        open={!!pendingDelete}
        onClose={() => setPendingDelete(null)}
        title="Remove document"
        footer={
          <>
            <Button variant="secondary" onClick={() => setPendingDelete(null)} disabled={isDeleting}>
              Cancel
            </Button>
            <Button variant="danger" onClick={confirmDelete} loading={isDeleting}>
              Delete
            </Button>
          </>
        }
      >
        Delete <span className="font-medium text-slate-800 dark:text-ink-primary">{pendingDelete?.original_filename}</span>?
        This permanently removes it from the server — its content will no longer be searchable in chat. This can't be undone.
      </Modal>

      {/* Opens at page 1 with no excerpt to highlight — unlike the citation
          flow in SourceReferences.jsx, there's no specific passage to jump
          to here, just "let me look at the document". Reuses the same
          modal so both entry points render/highlight identically. */}
      <PdfPreviewModal
        open={!!previewDoc}
        onClose={() => setPreviewDoc(null)}
        documentId={previewDoc?.document_id}
        pageNumber={1}
        title={previewDoc?.original_filename}
      />

      <ImageGalleryModal
        open={!!galleryDoc}
        onClose={() => setGalleryDoc(null)}
        documentId={galleryDoc?.document_id}
        title={galleryDoc ? `${galleryDoc.original_filename} — extracted images` : undefined}
      />
    </div>
  )
}
