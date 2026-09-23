import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Activity,
  Camera,
  CheckCircle2,
  Eye,
  EyeOff,
  Layers,
  ShieldCheck,
  UserPlus,
} from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Logo from '../components/ui/Logo'
import useAuth from '../hooks/useAuth'
import getErrorMessage from '../utils/errorMessage'

const HIGHLIGHTS = [
  {
    icon: Camera,
    title: '38-Class Hybrid Deep Vision Network',
    description: 'CBAM attention and EfficientNet feature fusion for real-time plant leaf disease classification.',
  },
  {
    icon: Layers,
    title: 'Reciprocal Rank Fusion (RRF) Retrieval',
    description: 'Dense neural embeddings fused with BM25 lexical search over 749 verified agronomy vectors.',
  },
  {
    icon: ShieldCheck,
    title: 'Private & Tenant-Isolated Workspaces',
    description: 'Each tenant has segregated document collections and cryptographic token validation.',
  },
]

export default function Signup() {
  const { signup } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [consent, setConsent] = useState(false)
  const [error, setError] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleSubmit(event) {
    event.preventDefault()
    setError(null)

    if (!consent) {
      setError('You must agree to account data storage to proceed.')
      return
    }

    setIsSubmitting(true)
    try {
      await signup(email, password, consent)
      navigate('/chat', { replace: true })
    } catch (err) {
      setError(getErrorMessage(err, 'Could not create your account.'))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="grid min-h-screen w-full lg:grid-cols-12">
      {/* Left Column: Project Showcase & Value Proposition */}
      <div className="hidden flex-col justify-between border-r border-border-light bg-slate-900/[0.02] p-10 dark:border-border dark:bg-white/[0.015] lg:col-span-6 lg:flex xl:col-span-7 xl:p-14">
        <div>
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-border-light bg-slate-900/5 text-accent-600 dark:border-border dark:bg-white/[0.03] dark:text-accent-500">
              <Logo size={20} />
            </span>
            <div>
              <p className="font-display text-base font-bold text-slate-900 dark:text-ink-primary">AgroSense</p>
              <p className="text-xs text-slate-400 dark:text-ink-muted">AI Agronomy & Document Intelligence</p>
            </div>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="mt-12 max-w-lg space-y-4"
          >
            <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-600 dark:border-emerald-500/30 dark:bg-emerald-500/15 dark:text-emerald-400">
              <Activity size={13} />
              Zero-Cost Multi-Tenant RAG
            </span>
            <h1 className="font-display text-2xl font-bold tracking-tight text-slate-900 dark:text-ink-primary sm:text-3xl xl:text-4xl">
              Build Your Agricultural Knowledge Base & Diagnostic Hub
            </h1>
            <p className="text-sm leading-relaxed text-slate-500 dark:text-ink-muted">
              Get immediate access to progressive streaming leaf diagnoses, interactive spray dosage calculators, and
              grounded document search with university extension citations.
            </p>
          </motion.div>

          <div className="mt-10 space-y-4 max-w-lg">
            {HIGHLIGHTS.map(({ icon: Icon, title, description }, idx) => (
              <motion.div
                key={title}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.1 * (idx + 1) }}
                className="flex items-start gap-3 rounded-xl border border-border-light bg-white/50 p-3.5 backdrop-blur-sm dark:border-border dark:bg-white/[0.02]"
              >
                <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-border-light bg-slate-900/5 text-accent-600 dark:border-border dark:bg-white/[0.03] dark:text-accent-500">
                  <Icon size={16} strokeWidth={1.75} />
                </span>
                <div>
                  <h3 className="font-display text-xs font-semibold text-slate-800 dark:text-ink-primary">{title}</h3>
                  <p className="mt-0.5 text-[11px] leading-relaxed text-slate-500 dark:text-ink-muted">{description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="mt-12 flex flex-wrap items-center gap-4 border-t border-border-light pt-6 text-xs text-slate-400 dark:border-border dark:text-ink-muted">
          <span className="flex items-center gap-1.5">
            <CheckCircle2 size={13} className="text-emerald-500" />
            Instant Activation
          </span>
          <span className="flex items-center gap-1.5">
            <CheckCircle2 size={13} className="text-emerald-500" />
            100% Private Document Isolation
          </span>
          <span className="flex items-center gap-1.5">
            <CheckCircle2 size={13} className="text-emerald-500" />
            Groq Llama 3.3 70B & Local Vision
          </span>
        </div>
      </div>

      {/* Right Column: Registration Form */}
      <div className="flex flex-col justify-center px-6 py-12 lg:col-span-6 lg:px-12 xl:col-span-5 xl:px-16">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mx-auto w-full max-w-md space-y-6"
        >
          {/* Mobile branding header */}
          <div className="flex flex-col items-center text-center lg:hidden">
            <span className="mb-3 flex h-11 w-11 items-center justify-center rounded-xl border border-border-light bg-slate-900/5 text-accent-600 dark:border-border dark:bg-white/[0.03] dark:text-accent-500">
              <Logo size={22} />
            </span>
            <h2 className="font-display text-lg font-bold text-slate-900 dark:text-ink-primary">AgroSense</h2>
            <p className="text-xs text-slate-400 dark:text-ink-muted">AI Agronomy & Document Intelligence</p>
          </div>

          <div className="space-y-1 text-left">
            <h2 className="font-display text-2xl font-bold text-slate-900 dark:text-ink-primary">Create account</h2>
            <p className="text-sm text-slate-500 dark:text-ink-muted">
              Get started with your private agronomy workspace.
            </p>
          </div>

          <Card padding="lg" className="border-border-light shadow-sm dark:border-border">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label
                  htmlFor="email"
                  className="mb-1.5 block text-xs font-medium text-slate-600 dark:text-ink-secondary"
                >
                  Email address
                </label>
                <input
                  id="email"
                  type="email"
                  required
                  autoComplete="email"
                  placeholder="name@example.com"
                  className="input"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                />
              </div>

              <div>
                <label
                  htmlFor="password"
                  className="mb-1.5 block text-xs font-medium text-slate-600 dark:text-ink-secondary"
                >
                  Password
                </label>
                <div className="relative">
                  <input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    required
                    minLength={8}
                    autoComplete="new-password"
                    placeholder="At least 8 characters"
                    className="input pr-10"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((current) => !current)}
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 transition-colors hover:text-slate-600 dark:text-ink-muted dark:hover:text-ink-secondary"
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
                <p className="mt-1 text-[11px] text-slate-400 dark:text-ink-muted">Minimum 8 characters.</p>
              </div>

              <label className="flex items-start gap-2.5 text-xs text-slate-500 dark:text-ink-muted">
                <input
                  type="checkbox"
                  className="mt-0.5 h-3.5 w-3.5 rounded border-border-light dark:border-border"
                  checked={consent}
                  onChange={(event) => setConsent(event.target.checked)}
                />
                <span>
                  I agree that my account credentials and documents are stored securely in accordance with local tenant
                  privacy policies.
                </span>
              </label>

              {error && (
                <div className="rounded-lg border border-danger/20 bg-danger/10 p-2.5 text-xs text-danger">
                  {error}
                </div>
              )}

              <Button
                type="submit"
                variant="primary"
                size="md"
                icon={UserPlus}
                loading={isSubmitting}
                className="w-full justify-center"
              >
                Create workspace
              </Button>
            </form>
          </Card>

          <p className="text-center text-sm text-slate-500 dark:text-ink-muted">
            Already have an account?{' '}
            <Link to="/login" className="font-semibold text-accent-600 hover:underline dark:text-accent-400">
              Log in
            </Link>
          </p>
        </motion.div>
      </div>
    </div>
  )
}
