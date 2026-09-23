import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AlertCircle,
  AlertTriangle,
  BookOpen,
  Calendar,
  CheckCircle2,
  Clock,
  Droplets,
  ExternalLink,
  FileText,
  FlaskConical,
  Globe2,
  Info,
  Leaf,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Sprout,
  ThermometerSun,
} from 'lucide-react'
import CitedAnswer from '../chat/CitedAnswer'
import SourceReferences from '../chat/SourceReferences'
import {
  REGULATORY_JURISDICTIONS,
  getAgronomicGuide,
  getChemicalRegulatoryStatus,
} from '../../utils/agronomyData'

const TABS = [
  { id: 'overview', label: 'Overview & Symptoms', icon: BookOpen },
  { id: 'organic', label: 'Organic Remedies', icon: Leaf },
  { id: 'chemical', label: 'Chemical Control & Dosages', icon: FlaskConical },
  { id: 'prevention', label: 'Prevention Schedule', icon: Calendar },
  { id: 'sources', label: 'Verified Sources & PDF Citations', icon: FileText },
]

/**
 * 5-Tab Treatment Plan Hub providing structured agronomic management,
 * bio-remedies, chemical dosages, regional regulatory filtering, seasonal schedule, and RAG citations.
 */
export default function TreatmentPlanTabs({
  diagnosis,
  answer,
  sources = [],
  query,
  isStreaming = false,
}) {
  const [activeTab, setActiveTab] = useState('overview')
  const [jurisdiction, setJurisdiction] = useState('EPA')

  const crop = diagnosis?.crop || ''
  const disease = diagnosis?.disease || ''
  const guide = getAgronomicGuide(disease, crop)

  return (
    <div className="panel space-y-5 rounded-panel border border-border-light p-5 shadow-soft dark:border-border">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border-light pb-4 dark:border-border">
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-500/15 text-accent-600 dark:text-accent-400">
            <Sparkles size={16} strokeWidth={2} />
          </span>
          <h3 className="font-display text-lg font-bold text-slate-900 dark:text-ink-primary">
            Agronomic Treatment & Management Plan
          </h3>
        </div>
        <span className="text-xs text-slate-500 dark:text-ink-muted">
          Pathogen: <strong className="font-medium text-slate-700 dark:text-ink-secondary">{guide.pathogen}</strong>
        </span>
      </div>

      {/* Tab Navigation Buttons */}
      <div
        role="tablist"
        aria-label="Treatment plan tabs"
        className="flex flex-wrap gap-1.5 rounded-xl border border-border-light bg-slate-900/[0.02] p-1.5 dark:border-border dark:bg-white/[0.02]"
      >
        {TABS.map((tab) => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          const isSourcesTab = tab.id === 'sources'

          return (
            <button
              key={tab.id}
              role="tab"
              id={`tab-${tab.id}`}
              aria-controls={`tabpanel-${tab.id}`}
              aria-selected={isActive}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold transition-all ${
                isActive
                  ? 'bg-accent-500 text-white shadow-sm dark:bg-accent-500'
                  : 'text-slate-600 hover:bg-slate-900/5 hover:text-slate-900 dark:text-ink-secondary dark:hover:bg-white/5 dark:hover:text-ink-primary'
              }`}
            >
              <Icon size={14} className={isActive ? 'text-white' : 'text-slate-400 dark:text-ink-muted'} />
              <span>{tab.label}</span>
              {isSourcesTab && sources?.length > 0 && (
                <span
                  className={`rounded-full px-1.5 py-0.2 text-[10px] font-bold ${
                    isActive ? 'bg-white/20 text-white' : 'bg-accent-500/15 text-accent-600 dark:text-accent-400'
                  }`}
                >
                  {sources.length}
                </span>
              )}
            </button>
          )
        })}
      </div>

      {/* Tab Panels */}
      <div className="pt-2">
        <AnimatePresence initial={false}>
          {/* TAB 1: OVERVIEW & SYMPTOMS */}
          {activeTab === 'overview' && (
            <motion.div
              key="overview-tab"
              id="tabpanel-overview"
              role="tabpanel"
              aria-labelledby="tab-overview"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="space-y-5"
            >
              {/* Primary Visual Symptoms */}
              <div className="space-y-2">
                <h4 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-ink-muted">
                  <Info size={14} className="text-accent-500" />
                  Primary Diagnostic Symptoms
                </h4>
                <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                  {guide.symptoms.map((symptom, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-2.5 rounded-xl border border-border-light bg-white/60 p-3 text-xs leading-relaxed text-slate-700 dark:border-border dark:bg-white/[0.02] dark:text-ink-secondary"
                    >
                      <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-accent-500/15 text-[10px] font-bold text-accent-600 dark:text-accent-400">
                        {idx + 1}
                      </span>
                      <span>{symptom}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* AI Diagnosis Explanation from RAG */}
              <div className="rounded-xl border border-border-light bg-slate-900/[0.02] p-4 dark:border-border dark:bg-white/[0.02]">
                <h4 className="mb-2 flex items-center justify-between font-display text-sm font-semibold text-slate-900 dark:text-ink-primary">
                  <span className="flex items-center gap-2">
                    <Sparkles size={14} className="text-accent-500" />
                    AI Agricultural Analysis & Grounded Context
                  </span>
                  {isStreaming && (
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-accent-500/10 px-2.5 py-0.5 text-[10px] font-semibold text-accent-600 dark:text-accent-400">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-500" />
                      Streaming response...
                    </span>
                  )}
                </h4>
                <div className="text-sm leading-relaxed text-slate-700 dark:text-ink-secondary">
                  {isStreaming ? (
                    <p className="whitespace-pre-wrap leading-relaxed">
                      {answer || 'Generating treatment plan...'}
                      <span className="ml-0.5 inline-block h-3.5 w-[2px] animate-pulse bg-accent-500 align-middle" />
                    </p>
                  ) : (
                    <CitedAnswer text={answer || 'Diagnosis completed. Review the treatment recommendations below.'} sources={sources} query={query} />
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {/* TAB 2: ORGANIC REMEDIES */}
          {activeTab === 'organic' && (
            <motion.div
              key="organic-tab"
              id="tabpanel-organic"
              role="tabpanel"
              aria-labelledby="tab-organic"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="space-y-4"
            >
              {/* Bio-Fungicides */}
              <div className="space-y-2">
                <h4 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-400">
                  <ShieldCheck size={14} />
                  Biological Fungicides & Antagonists
                </h4>
                <div className="space-y-2">
                  {guide.organicRemedies.bioFungicides.map((item, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-2.5 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-3 text-xs leading-relaxed text-slate-700 dark:border-emerald-500/20 dark:bg-emerald-500/10 dark:text-ink-secondary"
                    >
                      <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-emerald-600 dark:text-emerald-400" />
                      <span>{item}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Botanical Oils & Bio Extracts */}
              <div className="space-y-2">
                <h4 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-ink-muted">
                  <Droplets size={14} className="text-accent-500" />
                  Botanical Extracts & Natural Sprays
                </h4>
                <div className="space-y-2">
                  {guide.organicRemedies.botanicalOils.map((item, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-2.5 rounded-xl border border-border-light bg-white/60 p-3 text-xs leading-relaxed text-slate-700 dark:border-border dark:bg-white/[0.02] dark:text-ink-secondary"
                    >
                      <Leaf size={14} className="mt-0.5 shrink-0 text-accent-500" />
                      <span>{item}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Cultural Sanitation */}
              <div className="space-y-2">
                <h4 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-ink-muted">
                  <Sprout size={14} className="text-amber-500" />
                  Cultural Sanitation & Field Hygiene
                </h4>
                <div className="space-y-2">
                  {guide.organicRemedies.culturalPractices.map((item, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-2.5 rounded-xl border border-border-light bg-white/60 p-3 text-xs leading-relaxed text-slate-700 dark:border-border dark:bg-white/[0.02] dark:text-ink-secondary"
                    >
                      <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-accent-500" />
                      <span>{item}</span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* TAB 3: CHEMICAL CONTROL & DOSAGES */}
          {activeTab === 'chemical' && (
            <motion.div
              key="chemical-tab"
              id="tabpanel-chemical"
              role="tabpanel"
              aria-labelledby="tab-chemical"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="space-y-4"
            >
              {/* Regional Chemical Regulation Filter Bar */}
              <div className="rounded-xl border border-border-light bg-slate-900/[0.02] p-3.5 dark:border-border dark:bg-white/[0.02]">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 dark:text-ink-secondary">
                    <Globe2 size={15} className="text-accent-500" />
                    <span>Regional Regulatory Jurisdiction:</span>
                  </div>

                  {/* Jurisdiction Selector Dropdown / Buttons */}
                  <div className="flex flex-wrap items-center gap-1.5">
                    <select
                      id="regulatory-jurisdiction-select"
                      data-testid="regulatory-jurisdiction-select"
                      value={jurisdiction}
                      onChange={(e) => setJurisdiction(e.target.value)}
                      className="rounded-lg border border-border-light bg-white px-3 py-1.5 text-xs font-medium text-slate-800 shadow-sm outline-none transition-colors focus:border-accent-500 focus:ring-1 focus:ring-accent-500 dark:border-border dark:bg-card dark:text-ink-primary"
                    >
                      {REGULATORY_JURISDICTIONS.map((reg) => (
                        <option key={reg.id} value={reg.id}>
                          {reg.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Regulation Description / Summary Notice */}
                <div className="mt-2.5 text-[11px] text-slate-500 dark:text-ink-muted">
                  {jurisdiction === 'EFSA' && (
                    <p className="flex items-center gap-1.5 text-amber-700 dark:text-amber-300">
                      <AlertTriangle size={13} className="shrink-0" />
                      <span>
                        <strong>European Union (EFSA) Active:</strong> Substances like Chlorothalonil and Mancozeb are non-renewed or phased-out in the EU. Compliant alternatives are badged in green.
                      </span>
                    </p>
                  )}
                  {jurisdiction === 'OMRI' && (
                    <p className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-300">
                      <Leaf size={13} className="shrink-0" />
                      <span>
                        <strong>Global Organic (OMRI Only) Active:</strong> Synthetic chemicals are prohibited. Refer to Tab 2 (Organic Remedies) for certified biological controls.
                      </span>
                    </p>
                  )}
                  {jurisdiction === 'EPA' && (
                    <p>
                      <strong>USA (EPA FIFRA) Active:</strong> Standard label rates and worker protection standards apply.
                    </p>
                  )}
                  {jurisdiction === 'CIBRC' && (
                    <p>
                      <strong>India (CIBRC) Active:</strong> Formulations registered under the Insecticides Act, 1968.
                    </p>
                  )}
                </div>
              </div>

              {/* Active Ingredients Table */}
              <div className="space-y-2">
                <h4 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-ink-muted">
                  <FlaskConical size={14} className="text-accent-500" />
                  Recommended Active Chemical Ingredients & Dosages
                </h4>
                <div className="overflow-x-auto rounded-xl border border-border-light dark:border-border">
                  <table className="w-full text-left text-xs">
                    <thead className="border-b border-border-light bg-slate-900/[0.03] text-slate-500 dark:border-border dark:bg-white/[0.03] dark:text-ink-muted">
                      <tr>
                        <th className="px-3.5 py-2.5 font-semibold">Active Formulation</th>
                        <th className="px-3.5 py-2.5 font-semibold">Regulatory Status</th>
                        <th className="px-3.5 py-2.5 font-semibold">Dilution Rate</th>
                        <th className="px-3.5 py-2.5 font-semibold">Per Acre Rate</th>
                        <th className="px-3.5 py-2.5 font-semibold">Pre-Harvest (PHI)</th>
                        <th className="px-3.5 py-2.5 font-semibold">Re-Entry (REI)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border-light dark:divide-border">
                      {guide.chemicalControl.activeIngredients.map((item, idx) => {
                        const regStatus = getChemicalRegulatoryStatus(item.name, jurisdiction)
                        return (
                          <tr
                            key={idx}
                            className={`hover:bg-slate-900/[0.01] dark:hover:bg-white/[0.01] ${
                              regStatus.isRestricted ? 'bg-rose-500/[0.03] dark:bg-rose-500/[0.05]' : ''
                            }`}
                          >
                            <td className="px-3.5 py-2.5 font-medium text-slate-800 dark:text-ink-primary">
                              {item.name}
                            </td>
                            <td className="px-3.5 py-2.5">
                              <span
                                className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${
                                  regStatus.badgeClass ||
                                  (regStatus.isRestricted
                                    ? 'border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300'
                                    : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300')
                                }`}
                              >
                                {regStatus.isRestricted ? (
                                  <ShieldAlert size={11} className="shrink-0" />
                                ) : (
                                  <ShieldCheck size={11} className="shrink-0" />
                                )}
                                <span>{regStatus.badge}</span>
                              </span>
                            </td>
                            <td className="px-3.5 py-2.5 font-mono text-accent-600 dark:text-accent-400">
                              {item.rate}
                            </td>
                            <td className="px-3.5 py-2.5 text-slate-600 dark:text-ink-secondary">
                              {item.perAcre}
                            </td>
                            <td className="px-3.5 py-2.5 text-slate-600 dark:text-ink-secondary">
                              {item.phi}
                            </td>
                            <td className="px-3.5 py-2.5 text-slate-600 dark:text-ink-secondary">
                              {item.rei}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Spray Interval & Resistance Management */}
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-border-light bg-white/60 p-3.5 dark:border-border dark:bg-white/[0.02]">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-slate-800 dark:text-ink-primary">
                    <Clock size={13} className="text-accent-500" />
                    Application Frequency
                  </div>
                  <p className="text-xs text-slate-600 dark:text-ink-secondary">
                    {guide.chemicalControl.sprayInterval}
                  </p>
                </div>

                <div className="rounded-xl border border-border-light bg-white/60 p-3.5 dark:border-border dark:bg-white/[0.02]">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-slate-800 dark:text-ink-primary">
                    <Shield size={13} className="text-accent-500" />
                    FRAC Resistance Management
                  </div>
                  <p className="text-xs text-slate-600 dark:text-ink-secondary">
                    {guide.chemicalControl.resistanceManagement}
                  </p>
                </div>
              </div>

              {/* PPE & Safety Warning */}
              <div className="flex items-start gap-2.5 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-900 dark:border-amber-500/20 dark:bg-amber-500/10 dark:text-amber-200">
                <AlertCircle size={15} className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" />
                <span>
                  <strong>Safety First:</strong> Always wear certified Personal Protective Equipment (chemical-resistant gloves, eye protection, respirator) when mixing and applying synthetic pesticides. Observe label directions strictly.
                </span>
              </div>
            </motion.div>
          )}

          {/* TAB 4: PREVENTION SCHEDULE */}
          {activeTab === 'prevention' && (
            <motion.div
              key="prevention-tab"
              id="tabpanel-prevention"
              role="tabpanel"
              aria-labelledby="tab-prevention"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="space-y-4"
            >
              <h4 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-ink-muted">
                <Calendar size={14} className="text-accent-500" />
                4-Stage Seasonal Crop Protection Calendar
              </h4>

              <div className="relative space-y-3 pl-4 before:absolute before:bottom-2 before:left-1.5 before:top-2 before:w-0.5 before:bg-accent-500/20">
                {guide.preventionSchedule.map((stage, idx) => (
                  <div key={idx} className="relative rounded-xl border border-border-light bg-white/60 p-3.5 dark:border-border dark:bg-white/[0.02]">
                    <div className="absolute -left-[22px] top-4 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-accent-500 text-[9px] font-bold text-white">
                      {idx + 1}
                    </div>
                    <h5 className="font-semibold text-slate-800 dark:text-ink-primary">
                      {stage.stage}
                    </h5>
                    <p className="mt-1 text-xs text-slate-600 dark:text-ink-secondary">
                      {stage.actions}
                    </p>
                  </div>
                ))}
              </div>

              {/* Environmental Risk Advisory */}
              <div className="flex items-start gap-2.5 rounded-xl border border-border-light bg-slate-900/[0.02] p-3 text-xs text-slate-600 dark:border-border dark:bg-white/[0.02] dark:text-ink-secondary">
                <ThermometerSun size={15} className="mt-0.5 shrink-0 text-accent-500" />
                <span>
                  <strong>Microclimate Awareness:</strong> Foliar pathogens thrive in leaf wetness periods lasting &gt;6 hours at temperatures between 65°F and 85°F. Adjust scouting frequency after periods of sustained rain or dense dew.
                </span>
              </div>
            </motion.div>
          )}

          {/* TAB 5: VERIFIED SOURCES & CITATIONS */}
          {activeTab === 'sources' && (
            <motion.div
              key="sources-tab"
              id="tabpanel-sources"
              role="tabpanel"
              aria-labelledby="tab-sources"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="space-y-4"
            >
              <div className="flex items-center justify-between">
                <h4 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-ink-muted">
                  <FileText size={14} className="text-accent-500" />
                  RAG Retrieved Documents & Citations ({sources?.length || 0})
                </h4>
              </div>

              {sources?.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs text-slate-500 dark:text-ink-muted">
                    The treatment recommendations above are grounded in the following uploaded agronomy documents and university manuals:
                  </p>
                  <SourceReferences sources={sources} query={query} />
                </div>
              ) : (
                <div className="rounded-xl border border-border-light bg-slate-900/[0.02] p-6 text-center text-xs text-slate-500 dark:border-border dark:bg-white/[0.02] dark:text-ink-muted">
                  <BookOpen size={24} className="mx-auto mb-2 text-slate-400" />
                  <p className="font-semibold text-slate-700 dark:text-ink-secondary">
                    No custom user documents cited
                  </p>
                  <p className="mt-1">
                    This diagnosis used LeafSense deep vision classification and built-in university agronomic protocols. Upload agricultural PDFs in the Documents tab for customized corpus retrieval!
                  </p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
