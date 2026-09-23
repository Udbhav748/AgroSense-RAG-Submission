import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight, Camera, FileText, MessageCircle, UploadCloud } from 'lucide-react'
import Card from '../components/ui/Card'
import Logo from '../components/ui/Logo'
import { getHealthStatus } from '../services/healthService'

const ACTIONS = [
  {
    to: '/diagnose',
    icon: Camera,
    title: 'Plant Leaf Diagnosis',
    description: 'Diagnose 38 crop disease classes via deep vision and get grounded treatment plans.',
    badge: 'Vision AI',
  },
  {
    to: '/chat',
    icon: MessageCircle,
    title: 'Start a Conversation',
    description: 'Ask questions grounded in agricultural knowledge and uploaded documents.',
  },
  {
    to: '/documents',
    icon: FileText,
    title: 'Browse Documents',
    description: 'Manage crop guides, fact sheets, and indexed knowledge files.',
  },
  {
    to: '/upload',
    icon: UploadCloud,
    title: 'Upload Documents',
    description: 'Add new PDF documents to the vector knowledge base in seconds.',
  },
]

export default function Home() {
  const [status, setStatus] = useState('checking')
  const [provider, setProvider] = useState('')

  useEffect(() => {
    getHealthStatus()
      .then((health) => {
        setStatus('online')
        setProvider(health?.llm?.provider ?? '')
      })
      .catch(() => setStatus('offline'))
  }, [])

  return (
    <div className="flex flex-col items-center py-10 text-center sm:py-16">
      <motion.span
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-5 flex h-12 w-12 items-center justify-center rounded-xl border border-border-light bg-slate-900/5 text-accent-600 dark:border-border dark:bg-white/[0.03] dark:text-accent-500"
      >
        <Logo size={24} />
      </motion.span>

      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="font-display text-3xl font-bold text-slate-900 dark:text-ink-primary sm:text-4xl"
      >
        AgroSense Multimodal Intelligence
      </motion.h1>

      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mt-3 max-w-lg text-sm text-slate-500 dark:text-ink-muted"
      >
        Deep learning plant leaf pathology and grounded document RAG with university extension citations.
      </motion.p>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15 }}
        className="mt-6 inline-flex items-center gap-2 rounded-full border border-border-light bg-white/60 px-4 py-1.5 text-xs text-slate-500 dark:border-border dark:bg-white/[0.03] dark:text-ink-muted"
      >
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            status === 'online' ? 'bg-success' : status === 'offline' ? 'bg-danger' : 'bg-warning animate-pulse-soft'
          }`}
        />
        {status === 'checking' && 'Checking backend...'}
        {status === 'online' && (provider ? `Backend connected · ${provider}` : 'Backend connected')}
        {status === 'offline' && 'Backend unreachable'}
      </motion.div>

      <div className="mt-10 grid w-full max-w-4xl gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {ACTIONS.map(({ to, icon: Icon, title, description, badge }) => (
          <Card key={to} as={Link} to={to} hover padding="lg" className="group flex flex-col items-start text-left">
            <div className="mb-4 flex w-full items-center justify-between">
              <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-border-light bg-slate-900/5 text-slate-500 dark:border-border dark:bg-white/[0.03] dark:text-ink-secondary">
                <Icon size={17} strokeWidth={1.5} />
              </span>
              {badge && (
                <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-600 dark:border-emerald-500/30 dark:bg-emerald-500/15 dark:text-emerald-400">
                  {badge}
                </span>
              )}
            </div>
            <h3 className="font-display text-sm font-semibold text-slate-800 dark:text-ink-primary">{title}</h3>
            <p className="mt-1 text-xs text-slate-500 dark:text-ink-muted">{description}</p>
            <span className="mt-4 flex items-center gap-1 text-xs font-medium text-accent-600 opacity-0 transition-opacity group-hover:opacity-100 dark:text-accent-500">
              Get started <ArrowRight size={12} />
            </span>
          </Card>
        ))}
      </div>
    </div>
  )
}
