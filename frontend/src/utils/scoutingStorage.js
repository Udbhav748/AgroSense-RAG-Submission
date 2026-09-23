/**
 * Field Scouting Log & Outbreak History Storage Utility
 * Manages client-side persistent storage, default agronomy logs,
 * real-time event broadcasting, and CSV/JSON export routines.
 */

export const SCOUTING_STORAGE_KEY = 'insightai_field_scouting_history'
export const SCOUTING_EVENT_NAME = 'insightai:scouting-updated'

export const DEFAULT_SCOUTING_LOGS = [
  {
    id: 'scout-001',
    timestamp: '2026-08-14T14:30:00.000Z',
    crop: 'Tomato',
    disease: 'Early Blight',
    severity: 'Severe',
    confidence: 0.94,
    location: 'North Orchard Block B',
    notes: 'Concentric target ring spots detected on lower foliage. Spreading along drip line.',
    remedyApplied: 'Copper Hydroxide 2.5g/L',
  },
  {
    id: 'scout-002',
    timestamp: '2026-08-12T09:15:00.000Z',
    crop: 'Potato',
    disease: 'Late Blight',
    severity: 'Severe',
    confidence: 0.91,
    location: 'West Field Plot 4',
    notes: 'Water-soaked lesions on leaf margins with white sporulation on undersides.',
    remedyApplied: 'Mancozeb 75% WP',
  },
  {
    id: 'scout-003',
    timestamp: '2026-08-10T16:45:00.000Z',
    crop: 'Corn',
    disease: 'Common Rust',
    severity: 'Moderate',
    confidence: 0.88,
    location: 'South Ridge Section 2',
    notes: 'Small cinnamon-brown pustules on upper and lower surfaces.',
    remedyApplied: 'Azoxystrobin 250 SC',
  },
  {
    id: 'scout-004',
    timestamp: '2026-08-08T11:20:00.000Z',
    crop: 'Tomato',
    disease: 'Healthy',
    severity: 'Low',
    confidence: 0.98,
    location: 'Greenhouse 1 Bay A',
    notes: 'Vigorous vegetative growth, clear leaf canopy, no pathogen indicators.',
    remedyApplied: 'None - Preventative Trichoderma',
  },
  {
    id: 'scout-005',
    timestamp: '2026-08-05T08:00:00.000Z',
    crop: 'Apple',
    disease: 'Apple Scab',
    severity: 'Moderate',
    confidence: 0.87,
    location: 'East Slope Row 12',
    notes: 'Olive-green velvety lesions on leaves and young fruit.',
    remedyApplied: 'Captan 50 WP',
  },
]

/**
 * Retrieve scouting logs from localStorage or fallback to initial realistic records.
 */
export function getScoutingLogs() {
  if (typeof window === 'undefined') return DEFAULT_SCOUTING_LOGS
  try {
    const raw = window.localStorage?.getItem(SCOUTING_STORAGE_KEY)
    if (!raw) {
      // Seed default logs if key does not exist yet
      window.localStorage?.setItem(SCOUTING_STORAGE_KEY, JSON.stringify(DEFAULT_SCOUTING_LOGS))
      return DEFAULT_SCOUTING_LOGS
    }
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : DEFAULT_SCOUTING_LOGS
  } catch (err) {
    console.error('Failed to read scouting logs from localStorage:', err)
    return DEFAULT_SCOUTING_LOGS
  }
}

/**
 * Save array of scouting entries to localStorage and dispatch custom update event.
 */
export function saveScoutingLogs(logs) {
  if (typeof window === 'undefined') return logs
  try {
    window.localStorage?.setItem(SCOUTING_STORAGE_KEY, JSON.stringify(logs))
    window.dispatchEvent(new CustomEvent(SCOUTING_EVENT_NAME, { detail: logs }))
  } catch (err) {
    console.error('Failed to save scouting logs to localStorage:', err)
  }
  return logs
}

/**
 * Add a new scouting record.
 */
export function addScoutingEntry(entry) {
  const currentLogs = getScoutingLogs()
  const newEntry = {
    id: entry.id || `scout-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    timestamp: entry.timestamp || new Date().toISOString(),
    crop: entry.crop ? entry.crop.charAt(0).toUpperCase() + entry.crop.slice(1) : 'Crop',
    disease: entry.disease ? entry.disease.charAt(0).toUpperCase() + entry.disease.slice(1) : 'Unknown Condition',
    severity: entry.severity || 'Moderate',
    confidence: typeof entry.confidence === 'number' ? entry.confidence : 0.95,
    location: entry.location || 'North Field Sector 1',
    notes: entry.notes || '',
    remedyApplied: entry.remedyApplied || 'Pending Assessment',
  }

  const updated = [newEntry, ...currentLogs]
  saveScoutingLogs(updated)
  return newEntry
}

/**
 * Delete a scouting record by id.
 */
export function deleteScoutingEntry(id) {
  const currentLogs = getScoutingLogs()
  const updated = currentLogs.filter((item) => item.id !== id)
  saveScoutingLogs(updated)
  return updated
}

/**
 * Reset scouting records to default demonstration data.
 */
export function resetScoutingLogs() {
  saveScoutingLogs(DEFAULT_SCOUTING_LOGS)
  return DEFAULT_SCOUTING_LOGS
}

/**
 * Clear all scouting records.
 */
export function clearAllScoutingLogs() {
  saveScoutingLogs([])
  return []
}

/**
 * Export scouting entries to standard RFC-4180 CSV format and trigger browser download.
 */
export function exportScoutingToCSV(logs = getScoutingLogs(), filename) {
  const headers = ['id', 'timestamp', 'crop', 'disease', 'severity', 'confidence', 'location', 'notes', 'remedyApplied']
  
  const escapeCsv = (val) => {
    if (val == null) return '""'
    const str = String(val).replace(/"/g, '""')
    return `"${str}"`
  }

  const rows = logs.map((log) => [
    escapeCsv(log.id),
    escapeCsv(log.timestamp),
    escapeCsv(log.crop),
    escapeCsv(log.disease),
    escapeCsv(log.severity),
    escapeCsv(typeof log.confidence === 'number' ? log.confidence.toFixed(2) : log.confidence),
    escapeCsv(log.location),
    escapeCsv(log.notes),
    escapeCsv(log.remedyApplied),
  ].join(','))

  const csvContent = [headers.join(','), ...rows].join('\r\n')
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename || `Field_Scouting_History_${new Date().toISOString().slice(0, 10)}.csv`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

/**
 * Export scouting entries to formatted JSON format and trigger browser download.
 */
export function exportScoutingToJSON(logs = getScoutingLogs(), filename) {
  const jsonContent = JSON.stringify(logs, null, 2)
  const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename || `Field_Scouting_History_${new Date().toISOString().slice(0, 10)}.json`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}
