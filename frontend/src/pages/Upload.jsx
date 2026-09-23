import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowRight, FileText } from 'lucide-react'
import UploadZone from '../components/upload/UploadZone'
import PDFCard from '../components/upload/PDFCard'
import Button from '../components/ui/Button'
import Card from '../components/ui/Card'
import { getUploadHistory } from '../services/documentService'
import useUpload from '../hooks/useUpload'

export default function Upload() {
  const { file, progress, status, errorMessage, selectFile, reset, startUpload } = useUpload()
  const [collection, setCollection] = useState('')
  const [recent, setRecent] = useState([])
  const navigate = useNavigate()

  useEffect(() => {
    setRecent(getUploadHistory().slice(0, 3))
  }, [status])

  const handleSelect = (selected) => {
    selectFile(selected)
  }

  useEffect(() => {
    if (file && status === 'idle') {
      startUpload(collection.trim() || undefined)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file])

  return (
    <div className="mx-auto max-w-2xl space-y-6 py-4">
      <div className="text-center">
        <h2 className="font-display text-2xl font-bold">Upload a document</h2>
        <p className="mt-1.5 text-sm text-slate-500 dark:text-ink-muted">
          Add a PDF to your knowledge base — it'll be chunked, embedded, and ready to chat with in moments.
        </p>
      </div>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-center">
        <input
          value={collection}
          onChange={(event) => setCollection(event.target.value)}
          disabled={!!file}
          placeholder="Collection (optional) — e.g. finance, research"
          className="input w-full py-2.5 text-sm sm:max-w-md"
        />
        <p className="text-xs text-slate-400 dark:text-ink-muted sm:max-w-[16rem]">
          Tag this document into a named collection so a chat can be scoped to it.
        </p>
      </div>

      <AnimatePresence mode="wait">
        {!file ? (
          <motion.div key="dropzone" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <UploadZone onFileSelected={handleSelect} />
          </motion.div>
        ) : (
          <motion.div key="preview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-4">
            <PDFCard file={file} progress={progress} status={status} errorMessage={errorMessage} onRemove={reset} />

            {status === 'success' && (
              <div className="flex justify-center gap-3">
                <Button variant="secondary" onClick={reset}>
                  Upload another
                </Button>
                <Button variant="primary" icon={ArrowRight} iconPosition="right" onClick={() => navigate('/chat')}>
                  Start chatting
                </Button>
              </div>
            )}

            {status === 'error' && (
              <div className="flex justify-center">
                <Button variant="secondary" onClick={() => startUpload(collection.trim() || undefined)}>
                  Try again
                </Button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {recent.length > 0 && (
        <Card padding="md">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-700 dark:text-ink-secondary">Recent uploads</h3>
            <button
              type="button"
              onClick={() => navigate('/documents')}
              className="text-xs font-medium text-accent-600 hover:underline dark:text-accent-500"
            >
              View all
            </button>
          </div>
          <ul className="space-y-2">
            {recent.map((doc) => (
              <li
                key={doc.document_id}
                className="flex items-center gap-3 rounded-lg px-2 py-1.5 text-sm text-slate-600 dark:text-ink-secondary"
              >
                <FileText size={15} className="shrink-0 text-slate-400 dark:text-ink-muted" />
                <span className="truncate">{doc.original_filename}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  )
}
